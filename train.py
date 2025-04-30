import os
import numpy as np
import torch
import torchvision
import json
import argparse
import yaml
import shutil
from typing import Optional, Literal
from tqdm import tqdm
import concurrent.futures
import logging
from pydantic import BaseModel, field_validator, Field
from scene.cameras import Camera
from scene.dataset import parse_dataset, DatasetNames
from scene.model import parse_model, ModelNames
from utils.system_utils import searchForMaxIteration
from utils.renderer_utils import (
    parse_renderer,
    RendererNames,
    is_forward_only,
    is_support_vspace,
)
from utils.camera_utils import camera_to_JSON
from utils.loss_utils import get_loss, ssim, l1_loss, psnr
from lpips import LPIPS


class TrainerParams(BaseModel):
    debug: bool
    device: str
    dataset_type: DatasetNames
    renderer: RendererNames
    forward_renderer: RendererNames
    model_type: ModelNames
    model_path: str
    load_iteration: Optional[int]
    saving_iterations: list[int]
    max_iterations: int = Field(..., gt=0)
    trainer_seed: int = 0
    lambda_dssim: float = 0.2
    method_mask_loss: Literal["none", "cover", "penalty"] = "none"
    parallel_load: bool
    num_workers_load: int = Field(..., ge=-1)
    model_params: dict
    dataset_params: dict

    @field_validator("device")
    def validate_device(cls, value):
        if not torch.cuda.is_available() and value == "cuda":
            raise ValueError("CUDA is not available")
        elif value not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        return value

    @field_validator("renderer")
    def validate_renderer(cls, value):
        if is_forward_only(value):
            raise ValueError(
                f"Renderer {value} is only supported for forward rendering"
            )
        return value

    @field_validator("saving_iterations")
    def validate_saving_iterations(cls, value):
        if not all(isinstance(i, int) and i > 0 for i in value):
            raise ValueError("All saving iterations must be positive integers")
        return value


class Trainer:
    def __init__(self, trainer_options: dict):
        self._trainer_params = TrainerParams.model_validate(trainer_options)

        logging.basicConfig(
            level=logging.INFO if not self._trainer_params.debug else logging.DEBUG
        )
        self._device = torch.device(self._trainer_params.device)

        self._GaussianRasterizer, self._GaussianRasterizationSettings = parse_renderer(
            self._trainer_params.renderer
        )
        self._ForwardGaussianRasterizer, self._ForwardGaussianRasterizationSettings = (
            parse_renderer(self._trainer_params.forward_renderer)
        )
        self._GaussianModel = parse_model(self._trainer_params.model_type)

        self._dataset = parse_dataset(self._trainer_params.dataset_type)(
            self._trainer_params.dataset_params, self._dataset_context
        )
        self._gaussians = self._GaussianModel(
            self._trainer_params.model_params, self._model_context
        )

        self.__lpips_alex = LPIPS(net="alex").to(self._device)
        self.__lpips_vgg = LPIPS(net="vgg").to(self._device)
        self.__progress_bar = None
        self.__ema_loss_for_log = None

    @property
    def model(self):
        return self._gaussians

    @property
    def dataset(self):
        return self._dataset

    @property
    def _dataset_context(self) -> dict:
        return {
            "device": self._trainer_params.device,
            "parallel_load": self._trainer_params.parallel_load,
        }

    @property
    def _model_context(self) -> dict:
        return {
            "device": self._trainer_params.device,
            "camera_extent": self._dataset.nerf_norm["radius"],
            "camera_count": len(self._dataset.train_cameras),
            "camera_id_count": len(
                set(
                    [
                        camera.camera_info.camera_id
                        for camera in self._dataset.train_cameras
                    ]
                )
            ),
            "max_iterations": self._trainer_params.max_iterations,
            "method_mask_loss": self._trainer_params.method_mask_loss,
            "is_render_support_vspace": is_support_vspace(
                self._trainer_params.renderer
            ),
        }

    def load_model(self):
        if self._trainer_params.load_iteration is None:
            os.makedirs(self._trainer_params.model_path, exist_ok=True)
            with open(self._dataset.ply_path, "rb") as src_file, open(
                os.path.join(self._trainer_params.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = list()
            cameras = list[Camera]()
            cameras.extend(self._dataset.train_cameras)
            cameras.extend(self._dataset.test_cameras)
            sorted_cameras = sorted(
                cameras,
                key=lambda x: (
                    x.camera_info.camera_id,
                    os.path.join(x.camera_info.image_folder, x.camera_info.image_name),
                ),
            )
            for id, cam in enumerate(sorted_cameras):
                json_cams.append(camera_to_JSON(id, cam._camera_info))
            with open(
                os.path.join(self._trainer_params.model_path, "cameras.json"), "w"
            ) as file:
                json.dump(json_cams, file, indent=2)
            with open(
                os.path.join(self._trainer_params.model_path, "cfg_args"), "w"
            ) as file:
                file.write(f"Namespace(sh_degree={self.model.sh_degree})")
            self._gaussians.init(self._dataset)
        else:
            active_iteration = (
                searchForMaxIteration(
                    os.path.join(self._trainer_params.model_path, "point_cloud")
                )
                if self._trainer_params.load_iteration == -1
                else self._trainer_params.load_iteration
            )
            self._gaussians.load(
                os.path.join(
                    self._trainer_params.model_path,
                    "point_cloud",
                    "iteration_" + str(active_iteration),
                    "point_cloud.ply",
                ),
                self._dataset,
            )

    def train_model(self):
        if self._trainer_params.parallel_load:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for camera in self._dataset.train_cameras:
                    futures.append(executor.submit(camera.load))

                for _ in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Loading images",
                ):
                    pass

        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before training")
        np.random.seed(self._trainer_params.trainer_seed)
        camera_indices = np.arange(len(self._dataset.train_cameras))
        np.random.shuffle(camera_indices)
        idx = 0
        self.__progress_bar = tqdm(
            range(self._trainer_params.max_iterations), desc="Training progress"
        )
        for iteration in range(self._trainer_params.max_iterations):
            selected_train_camera: Camera = self._dataset.train_cameras[
                camera_indices[idx]
            ]
            idx += 1
            if idx == len(camera_indices):
                idx = 0
                np.random.shuffle(camera_indices)
            with torch.no_grad():
                self._gaussians.iteration_start(
                    iteration, selected_train_camera, self._dataset
                )
            self._gaussians.optimizer.zero_grad(set_to_none=True)
            render_pkgs = self._gaussians.render(
                selected_train_camera,
                self._GaussianRasterizationSettings,
                self._GaussianRasterizer,
            )
            if self._trainer_params.method_mask_loss == "cover":
                loss = get_loss(
                    render_pkgs["rendered_image"] * selected_train_camera.image_mask,
                    selected_train_camera.image * selected_train_camera.image_mask,
                    self._trainer_params.lambda_dssim,
                )
            else:
                loss = get_loss(
                    render_pkgs["rendered_image"],
                    selected_train_camera.image,
                    self._trainer_params.lambda_dssim,
                )

            loss += self._gaussians.get_regularization_loss(
                camera=selected_train_camera, dataset=self._dataset
            )
            loss.backward()

            if iteration < self._trainer_params.max_iterations - 1:
                self._gaussians.optimizer.step()
            with torch.no_grad():
                self._gaussians.iteration_end(
                    iteration, selected_train_camera, self._dataset
                )
                if self._trainer_params.debug:
                    path = os.path.join(
                        self._trainer_params.model_path,
                        "rendered",
                        f"iteration_{iteration}.png",
                    )
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torchvision.utils.save_image(render_pkgs["rendered_image"], path)
                self.__ema_loss_for_log = (
                    (0.4 * loss.item() + 0.6 * self.__ema_loss_for_log)
                    if self.__ema_loss_for_log is not None
                    else loss.item()
                )
                if (iteration + 1) % 10 == 0:
                    self.__progress_bar.set_postfix(
                        {
                            "Loss": f"{self.__ema_loss_for_log:.{4}f}",
                            "Size": f"{len(self._gaussians) / 1000:.2f} K",
                            "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda') / 1024 ** 2:.2f} MB",
                        }
                    )
                    self.__progress_bar.update(10)
                if (iteration + 1) in self._trainer_params.saving_iterations:
                    self._gaussians.save(
                        os.path.join(
                            self._trainer_params.model_path,
                            "point_cloud",
                            "iteration_" + str(iteration + 1),
                            "point_cloud.ply",
                        )
                    )
                if (iteration % 1000) == 0:
                    self._gaussians.one_up_sh_degree()
        self.__progress_bar.close()

    def eval_model(self, eval_option: str):
        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before evaluation")

        if len(self._dataset.test_cameras) == 0:
            eval_option = "train"

        train_results = {}
        test_results = {}

        with torch.no_grad():
            if eval_option in ["all", "test"]:
                for test_cam in self._dataset.test_cameras:
                    results = self.test_model(test_cam, None)
                    test_results[test_cam.camera_info.image_name] = {
                        "l1": results[0],
                        "psnr": results[1],
                        "dssim": results[2],
                        "ssim": results[3],
                        "lpips_alex": results[4],
                        "lpips_vgg": results[5],
                    }
                avgs_test = self.__compute_avgs(test_results)
                self.__write_json(
                    avgs_test,
                    os.path.join(self._trainer_params.model_path, "test_avgs.json"),
                )
                self.__write_json(
                    test_results,
                    os.path.join(self._trainer_params.model_path, "test_results.json"),
                )

            if eval_option in ["all", "train"]:
                for train_cam in self._dataset.train_cameras:
                    results = self.test_model(train_cam, None)
                    train_results[train_cam.camera_info.image_name] = {
                        "l1": results[0],
                        "psnr": results[1],
                        "dssim": results[2],
                        "ssim": results[3],
                        "lpips_alex": results[4],
                        "lpips_vgg": results[5],
                    }
                avgs_train = self.__compute_avgs(train_results)
                self.__write_json(
                    avgs_train,
                    os.path.join(self._trainer_params.model_path, "train_avgs.json"),
                )
                self.__write_json(
                    train_results,
                    os.path.join(self._trainer_params.model_path, "train_results.json"),
                )

    def test_model(self, camera: Camera, save_path: str) -> float:
        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before testing")
        render_pkgs = self._gaussians.render_forward_only(
            camera,
            self._ForwardGaussianRasterizationSettings,
            self._ForwardGaussianRasterizer,
        )
        suffix = (
            f"_{camera.camera_info.timestamp}"
            if camera.camera_info.timestamp is not None
            else ""
        )
        save_path = (
            os.path.join(
                self._trainer_params.model_path,
                "rendered",
                f"{camera.camera_info.image_name.split('.')[0]}{suffix}.{camera.camera_info.image_name.split('.')[1]}",
            )
            if save_path is None
            else save_path
        )
        l1_value = l1_loss(render_pkgs["rendered_image"], camera.image).item()
        psnr_value = psnr(render_pkgs["rendered_image"], camera.image).item()
        ssim_value = ssim(render_pkgs["rendered_image"], camera.image).item()
        dssim_value = (1 - ssim_value) / 2
        lpips_alex_value = self.__lpips_alex(
            render_pkgs["rendered_image"], camera.image
        ).item()
        lpips_vgg_value = self.__lpips_vgg(
            render_pkgs["rendered_image"], camera.image
        ).item()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(render_pkgs["rendered_image"], save_path)
        return l1_value, psnr_value, dssim_value, ssim_value, lpips_alex_value, lpips_vgg_value

    def __compute_avgs(self, results_dict):
        return {
            "l1": np.mean([res["l1"] for res in results_dict.values()]),
            "psnr": np.mean([res["psnr"] for res in results_dict.values()]),
            "dssim": np.mean([res["dssim"] for res in results_dict.values()]),
            "ssim": np.mean([res["ssim"] for res in results_dict.values()]),
            "lpips_alex": np.mean([res["lpips_alex"] for res in results_dict.values()]),
            "lpips_vgg": np.mean([res["lpips_vgg"] for res in results_dict.values()]),
        }

    def __write_json(self, data, filename):
        with open(filename, "w") as file:
            json.dump(data, file, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/default.yaml")
    arg_parser.add_argument("--skip_train", action="store_false", dest="train")
    arg_parser.add_argument("--skip_eval", action="store_false", dest="eval")
    arg_parser.add_argument("--eval_option", type=str, default="all",
                            choices=["all", "train", "test"])
    args = arg_parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        model_path = config["TRAINER"]["model_path"]
        os.makedirs(model_path, exist_ok=True)
        if os.path.abspath(args.config) != os.path.abspath(os.path.join(model_path, "config.yaml")):
            shutil.copy(args.config, os.path.join(model_path, "config.yaml"))
    if not args.train and not config["TRAINER"]["load_iteration"]:
        config["TRAINER"]["load_iteration"] = -1
    trainer = Trainer(config["TRAINER"])
    trainer.load_model()
    if args.train:
        trainer.train_model()
    if args.eval:
        trainer.eval_model(args.eval_option)
