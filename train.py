import os
import numpy as np
import torch
import torchvision
import json
import argparse
import yaml
import shutil
from typing import Optional
from tqdm import tqdm
import logging
from pydantic import BaseModel, field_validator, Field
from scene.cameras import Camera
from scene.dataset import parse_dataset, DatasetNames
from scene.model import parse_model, parse_cfg_args, ModelNames
from utils.system_utils import searchForMaxIteration
from utils.renderer_utils import parse_renderer, RendererNames, is_forward_only
from utils.camera_utils import camera_to_JSON
from utils.loss_utils import get_loss, ssim

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
            raise ValueError(f"Renderer {value} is only supported for forward rendering")
        return value

    @field_validator("saving_iterations")
    def validate_saving_iterations(cls, value):
        if not all(isinstance(i, int) and i > 0 for i in value):
            raise ValueError("All saving iterations must be positive integers")
        return value

class Trainer:
    def __init__(self, trainer_options: dict):
        self._trainer_params = TrainerParams.model_validate(trainer_options)

        logging.basicConfig(level=logging.INFO if not self._trainer_params.debug else logging.DEBUG)
        self._device = torch.device(self._trainer_params.device)

        self._GaussianRasterizer, self._GaussianRasterizationSettings = parse_renderer(
            self._trainer_params.renderer
        )
        self._ForwardGaussianRasterizer, self._ForwardGaussianRasterizationSettings = (
            parse_renderer(self._trainer_params.forward_renderer)
        )
        self._GaussianModel = parse_model(self._trainer_params.model_type)

        self._dataset = parse_dataset(self._trainer_params.dataset_type)(self._trainer_params.dataset_params, self._dataset_context)
        self._gaussians = self._GaussianModel(self._trainer_params.model_params, self._model_context)

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
                key=lambda x: os.path.join(
                    x.camera_info.image_folder, x.camera_info.image_name
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
                file.write(parse_cfg_args(self._trainer_params.model_type))
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
            loss = get_loss(
                render_pkgs["rendered_image"] * selected_train_camera.image_mask,
                selected_train_camera.image,
                self._trainer_params.lambda_dssim,
            ) + self._gaussians.get_regularization_loss(
                camera=selected_train_camera, dataset=self._dataset
            )
            loss.backward()

            if iteration < self._trainer_params.max_iterations - 1:
                self._gaussians.optimizer.step()
            with torch.no_grad():
                self._gaussians.iteration_end(
                    iteration, selected_train_camera, self._dataset
                )
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
        self.__progress_bar.close()

    def eval_model(self):
        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before evaluation")
        if len(self._dataset.test_cameras) == 0:
            raise ValueError("No test cameras in dataset, please set is_eval to True")
        ssim_sum = 0.0
        with torch.no_grad():
            for test_camera in self._dataset.test_cameras:
                ssim_sum += self.test_model(test_camera, None)
        logging.info(f"SSIM: {ssim_sum / len(self._dataset.test_cameras)}")

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
        ssim_loss = ssim(render_pkgs["rendered_image"], camera.image)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(render_pkgs["rendered_image"], save_path)
        return ssim_loss


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/default.yaml")
    arg_parser.add_argument("--skip_train", action="store_false", dest="train")
    arg_parser.add_argument("--skip_eval", action="store_false", dest="eval")
    args = arg_parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        model_path = config["TRAINER"]["model_path"]
        os.makedirs(model_path, exist_ok=True)
        shutil.copy(args.config, os.path.join(model_path, "config.yaml"))
    trainer = Trainer(config["TRAINER"])
    trainer.load_model()
    if args.train:
        trainer.train_model()
    if args.eval:
        trainer.eval_model()
