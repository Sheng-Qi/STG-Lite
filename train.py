import os
import numpy as np
import torch
import torchvision
import json
import argparse
import yaml
import shutil
from tqdm import tqdm
import logging
from scene.cameras import Camera
from scene.dataset import parse_dataset
from scene.model import parse_model, parse_cfg_args
from utils.system_utils import searchForMaxIteration, parse_renderer
from utils.camera_utils import camera_to_JSON
from utils.loss_utils import get_loss, ssim


class Trainer:
    def __init__(self, trainer_options: dict, debug: bool = False):
        self._renderer: str = trainer_options["renderer"]
        self._forward_renderer: str = trainer_options["forward_renderer"]
        self._model_type: str = trainer_options["model_type"]
        self._model_path: str = trainer_options["model_path"]
        self._load_iteration: int = trainer_options["load_iteration"]
        self._saving_iterations: list[int] = trainer_options["saving_iterations"]
        self._iterations: int = trainer_options["iterations"]
        self._trainer_seed: int = trainer_options["trainer_seed"]
        self._lambda_dssim: float = trainer_options["lambda_dssim"]
        self._model_params: dict = trainer_options["model_params"]
        self._dataset_type: str = trainer_options["dataset_type"]
        self._dataset_params: dict = trainer_options["dataset_params"]
        self._debug = debug

        self._GaussianRasterizer, self._GaussianRasterizationSettings = parse_renderer(
            self._renderer
        )
        self._ForwardGaussianRasterizer, self._ForwardGaussianRasterizationSettings = (
            parse_renderer(self._forward_renderer)
        )
        self._GaussianModel = parse_model(self._model_type)
        self._progress_bar = None
        self._ema_loss_for_log = None

        self._preprocess_dataset_config()
        self._dataset = parse_dataset(self._dataset_type)(self._dataset_params)

        self._preprocess_model_config()
        self._gaussians = self._GaussianModel(self._model_params)

    @property
    def model(self):
        return self._gaussians

    @property
    def dataset(self):
        return self._dataset

    def load_model(self):
        if self._load_iteration is None:
            os.makedirs(self._model_path, exist_ok=True)
            with open(self._dataset.ply_path, "rb") as src_file, open(
                os.path.join(self._model_path, "input.ply"), "wb"
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
            with open(os.path.join(self._model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file, indent=2)
            with open(os.path.join(self._model_path, "cfg_args"), "w") as file:
                txt_content = parse_cfg_args(self._model_type)
                file.write(txt_content)
            self._gaussians.init(self._dataset)
        else:
            active_iteration = (
                searchForMaxIteration(os.path.join(self._model_path, "point_cloud"))
                if self._load_iteration == -1
                else self._load_iteration
            )
            self._gaussians.load(
                os.path.join(
                    self._model_path,
                    "point_cloud",
                    "iteration_" + str(active_iteration),
                    "point_cloud.ply",
                ),
                self._dataset,
            )

    def train_model(self):
        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before training")
        np.random.seed(self._trainer_seed)
        self._progress_bar = tqdm(range(self._iterations), desc="Training progress")
        for iteration in range(self._iterations):
            selected_train_camera: Camera = np.random.choice(
                self._dataset.train_cameras
            )
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
                self._lambda_dssim,
            ) + self._gaussians.get_regularization_loss(
                camera=selected_train_camera, dataset=self._dataset
            )
            loss.backward()

            if iteration < self._iterations - 1:
                self._gaussians.optimizer.step()
            with torch.no_grad():
                self._gaussians.iteration_end(
                    iteration, selected_train_camera, self._dataset
                )
            with torch.no_grad():
                self._ema_loss_for_log = (
                    (0.4 * loss.item() + 0.6 * self._ema_loss_for_log)
                    if self._ema_loss_for_log is not None
                    else loss.item()
                )
                if iteration % 10 == 0:
                    self._progress_bar.set_postfix(
                        {
                            "Loss": f"{self._ema_loss_for_log:.{4}f}",
                            "Size": f"{len(self._gaussians) / 1000:.2f} K",
                            "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda') / 1024 ** 2:.2f} MB",
                        }
                    )
                    self._progress_bar.update(10)
                if iteration in self._saving_iterations:
                    self._gaussians.save(
                        os.path.join(
                            self._model_path,
                            "point_cloud",
                            "iteration_" + str(iteration),
                            "point_cloud.ply",
                        )
                    )
        self._progress_bar.close()

    def eval_model(self):
        if len(self._gaussians) == 0:
            raise ValueError("Need to load model before evaluation")
        if len(self._dataset.test_cameras) == 0:
            raise ValueError("No test cameras in dataset, please set is_eval to True")
        ssim_sum = 0.0
        with torch.no_grad():
            for test_camera in self._dataset.test_cameras:
                render_pkgs = self._gaussians.render_forward_only(
                    test_camera,
                    self._ForwardGaussianRasterizationSettings,
                    self._ForwardGaussianRasterizer,
                )
                ssim_sum += ssim(render_pkgs["rendered_image"], test_camera.image)
                suffix = (
                    f"_{test_camera.camera_info.timestamp}"
                    if self._dataset_type == "technicolor"
                    else ""
                )
                save_path = os.path.join(
                    self._model_path,
                    "rendered",
                    f"{test_camera.camera_info.image_name.split('.')[0]}_{suffix}.{test_camera.camera_info.image_name.split('.')[1]}",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torchvision.utils.save_image(render_pkgs["rendered_image"], save_path)
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
                self._model_path,
                "rendered",
                f"{camera.camera_info.image_name.split('.')[0]}_{suffix}.{camera.camera_info.image_name.split('.')[1]}",
            )
            if save_path is None
            else save_path
        )
        ssim_loss = ssim(render_pkgs["rendered_image"], camera.image)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(render_pkgs["rendered_image"], save_path)
        return ssim_loss

    def _preprocess_dataset_config(self):
        self._dataset_params["device"] = self._model_params["device"]

    def _preprocess_model_config(self):
        self._model_params["cameras_extent"] = self._dataset.nerf_norm["radius"]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, default="configs/default.yaml")
    arg_parser.add_argument("--skip_train", action="store_false", dest="train")
    arg_parser.add_argument("--skip_eval", action="store_false", dest="eval")
    arg_parser.add_argument("--debug", action="store_true")
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        model_path = config["TRAINER"]["model_path"]
        os.makedirs(model_path, exist_ok=True)
        shutil.copy(args.config, os.path.join(model_path, "config.yaml"))
    trainer = Trainer(config["TRAINER"], debug=args.debug)
    trainer.load_model()
    if args.train:
        trainer.train_model()
    if args.eval:
        trainer.eval_model()
