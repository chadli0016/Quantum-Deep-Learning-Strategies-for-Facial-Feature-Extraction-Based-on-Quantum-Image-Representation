

import os

import urllib.request



url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

url2 = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"



save_path = "detector.tflite"

save_path2 = "face_landmarker.task"



# Download detector if missing

if not os.path.exists(save_path):

    print("Downloading detector model...")

    urllib.request.urlretrieve(url, save_path)

    print(f"Saved: {save_path}")

else:

    print(f"{save_path} already exists. Skipping download.")



# Download face landmarker if missing

if not os.path.exists(save_path2):

    print("Downloading face landmarker model...")

    urllib.request.urlretrieve(url2, save_path2)

    print(f"Saved: {save_path2}")

else:

    print(f"{save_path2} already exists. Skipping download.")









import math

import numpy as np

from typing import Union, Tuple, Optional

import cv2

import pandas as pd

from scipy.stats import skew, kurtosis

import face_recognition

import mediapipe as mp

from typing import Optional, Tuple,Union

from qiskit import QuantumCircuit

from qiskit.quantum_info import Statevector

from mediapipe.tasks import python

from mediapipe.tasks.python import vision





class face_crop:



    MARGIN = 10

    ROW_SIZE = 10

    FONT_SIZE = 1

    FONT_THICKNESS = 1

    TEXT_COLOR = (255, 0, 0)  # blue in BGR for OpenCV



    def __init__(self, model_path: str = "detector.tflite"):

        self.model_path = model_path



    def _normalized_to_pixel_coordinates(

        self,

        normalized_x: float,

        normalized_y: float,

        image_width: int,

        image_height: int

    ) -> Union[None, Tuple[int, int]]:

        """Convert normalized coordinates to pixel coordinates."""



        def is_valid_normalized_value(value: float) -> bool:

            return (0.0 <= value <= 1.0) or math.isclose(value, 0.0) or math.isclose(value, 1.0)



        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):

            return None



        x_px = min(math.floor(normalized_x * image_width), image_width - 1)

        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px



    def visualize(self, image: np.ndarray, detection_result) -> np.ndarray:

        """

        Draw bounding boxes and keypoints on the image.

        Expects image in RGB or BGR only for display purposes.

        """

        annotated_image = image.copy()

        height, width = image.shape[:2]



        for detection in detection_result.detections:



            bbox = detection.bounding_box

            start_point = (bbox.origin_x, bbox.origin_y)

            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)



            cv2.rectangle(

                annotated_image,

                start_point,

                end_point,

                self.TEXT_COLOR,

                2

            )



            for keypoint in detection.keypoints:

                keypoint_px = self._normalized_to_pixel_coordinates(

                    keypoint.x, keypoint.y, width, height

                )

                if keypoint_px is not None:

                    cv2.circle(annotated_image, keypoint_px, 2, (0, 255, 0), -1)



            category = detection.categories[0]

            category_name = category.category_name if category.category_name else "face"

            probability = round(category.score, 2)

            result_text = f"{category_name} ({probability})"



            text_location = (

                self.MARGIN + bbox.origin_x,

                self.MARGIN + self.ROW_SIZE + bbox.origin_y

            )



            cv2.putText(

                annotated_image,

                result_text,

                text_location,

                cv2.FONT_HERSHEY_PLAIN,

                self.FONT_SIZE,

                self.TEXT_COLOR,

                self.FONT_THICKNESS

            )



        return annotated_image



    def face_cropped(

        self,

        IMAGE_FILE: str,

        padding: float = 0.2,

        output_size: Optional[Tuple[int, int]] = (128, 128),

        draw: bool = False

    ) -> Optional[np.ndarray]:

        # AI OPTIMIZED (MEDIA PIPE HAS OVER 400 LANDMARK )

        """

        Detect first face and return cropped face image.



        Args:

            IMAGE_FILE: image path

            padding: extra margin around detected box

            output_size: resize cropped face to this size; set None to keep original crop size

            draw: whether to also show annotated image



        Returns:

            cropped face as numpy array (BGR), or None if no face detected

        """



        base_options = python.BaseOptions(model_asset_path=self.model_path)

        options = vision.FaceDetectorOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.IMAGE

        )



        # MediaPipe Tasks image

        mp_image = mp.Image.create_from_file(IMAGE_FILE)



        with vision.FaceDetector.create_from_options(options) as detector:

            detection_result = detector.detect(mp_image)



        # Original image for cropping with OpenCV

        img = cv2.imread(IMAGE_FILE)

        if img is None:

            raise ValueError(f"Could not read image: {IMAGE_FILE}")



        if not detection_result.detections:

            return None



        # Use first detected face

        detection = detection_result.detections[0]

        bbox = detection.bounding_box



        x1 = bbox.origin_x

        y1 = bbox.origin_y

        x2 = bbox.origin_x + bbox.width

        y2 = bbox.origin_y + bbox.height



        h, w = img.shape[:2]



        pad_x = int(bbox.width * padding)

        pad_y = int(bbox.height * padding)



        x1 = max(0, x1 - pad_x)

        y1 = max(0, y1 - pad_y)

        x2 = min(w, x2 + pad_x)

        y2 = min(h, y2 + pad_y)



        cropped = img[y1:y2, x1:x2]



        if cropped.size == 0:

            return None



        if output_size is not None:

            cropped = cv2.resize(cropped, output_size)



        if draw:

            image_copy = np.copy(mp_image.numpy_view())

            annotated_image = self.visualize(image_copy, detection_result)



            # mp_image is RGB; cv2.imshow expects BGR-looking display,

            # so convert before showing

            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            cv2.imshow("Annotated Face Detection", annotated_bgr)

            cv2.waitKey(0)

            cv2.destroyAllWindows()



        return cropped







import cv2

import numpy as np





class QIP:



    def __init__(self, name="QIPE"):

        self.name = name



    # =========================================

    # U = F.H D F

    # =========================================

    def unitary_fourier_phase_operator(

        self,

        img,

        tau=0.001,

        size=(160, 160)

    ):



        img = img.copy()



        img = cv2.resize(img, size)



        img = img.astype(np.float64)



        # =========================================

        # FFT

        # =========================================



        F_img = np.fft.fft2(img)



        h, w = img.shape



        u = np.fft.fftfreq(h)

        v = np.fft.fftfreq(w)



        U, V = np.meshgrid(

            u,

            v,

            indexing="ij"

        )



        # =========================================

        # D = phase operator

        # =========================================



        D = np.exp(

            -1j * tau * np.sin(10 * (U**2 + V**2))

        )



        # =========================================

        # U = F.H D F

        # =========================================



        transformed = np.fft.ifft2(

            D * F_img

        )



        # FIX:

        # use magnitude of complex result



        # result = np.abs(transformed)

        result = transformed

        return result



    # =========================================

    # QIPE

    # =========================================

    def QIPE(self, img):

        if img.ndim == 3:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = img.astype(np.float64)



        TOT = img.sum()



        QIPE = np.sqrt(

            img / (TOT + 1e-12)

        )



        return QIPE



    # =========================================

    # QIIP

    # =========================================

    def QIIP(self, img):



        qipe = self.QIPE(

            img=img

        )



        result = self.unitary_fourier_phase_operator(

            img=qipe,

            tau=10,

            size=img.shape[:2]

        )

        return result



    # =========================================

    # SWITCH

    # =========================================

    def method(self, img, name=None):



        if name is None:

            name = self.name



        match name:



            case "QIPE":

                return self.QIPE(

                    img

                )



            case "QIIP":

                return self.QIIP(

                    img

                )



            case _:

                return img















from scipy.stats import skew, kurtosis

import scipy.sparse as sp

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import models, transforms

from skimage.feature import hog

from PIL import Image













class feature_extraction:

    def __init__(self,name="ALEXnet"):

        self.name=name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

        self.model_alexnet.classifier = torch.nn.Sequential(*list(self.model_alexnet.classifier.children())[:-1])

        self.model_alexnet = self.model_alexnet.to(self.device)

        self.model_alexnet.eval()

        self.transform = transforms.Compose([

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[0.485, 0.456, 0.406],

                std=[0.229, 0.224, 0.225]

            )

        ])

    def ALEXnet(self,img):

        if isinstance(img, str):

            img = Image.open(img).convert("RGB")

        elif isinstance(img, np.ndarray):

            img = Image.fromarray(img).convert("RGB")

        else:

            img = img.convert("RGB")

        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():

            features = self.model_alexnet(x)

        return features.cpu().numpy().flatten()

    def HOG(self,img):

        if img.ndim == 3:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features, _ = hog(

            img,

            orientations=9,

            pixels_per_cell=(8, 8),

            cells_per_block=(2, 2),

            block_norm="L2-Hys",

            visualize=True,

            feature_vector=True

        )

        return features

    def QFSD(self,QIIP, n_radial=8, n_angular=8):

        """

        QFSD: Quantum Fourier Spectrum Descriptor



        Input:

            QIIP: complex-valued QIIP image/state

        Output:

            1D feature vector

        """



        Q = np.abs(QIIP)

        Phi = np.angle(QIIP)



        h, w = Q.shape



        y, x = np.indices((h, w))

        cy, cx = h // 2, w // 2



        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        r = r / (r.max() + 1e-12)



        theta = np.arctan2(y - cy, x - cx)

        theta = (theta + np.pi) / (2 * np.pi)



        features = []



        # Radial features

        for k in range(n_radial):

            mask = (r >= k / n_radial) & (r < (k + 1) / n_radial)



            features.append(Q[mask].mean())

            features.append(Q[mask].std())

            features.append(Phi[mask].mean())

            features.append(Phi[mask].std())



        # Angular features

        for j in range(n_angular):

            mask = (theta >= j / n_angular) & (theta < (j + 1) / n_angular)



            features.append(Q[mask].mean())

            features.append(Q[mask].std())

            features.append(Phi[mask].mean())

            features.append(Phi[mask].std())



        # Energy

        E_Q = np.sum(Q ** 2)



        # Entropy

        p = Q.flatten() / (Q.sum() + 1e-12)

        H_Q = -np.sum(p * np.log(p + 1e-12))



        features.append(E_Q)

        features.append(H_Q)



        return np.array(features, dtype=np.float64)

    def quantum_inspired(self,img):

        def quantum_patch_feature(patch):

            patch = patch.flatten() / 255.0

            patch = patch[:4]  # 4 qubits

            qc = QuantumCircuit(4)

            for i, v in enumerate(patch):

                qc.ry(np.pi*v, i)

            qc.cx(0,1); qc.cx(1,2); qc.cx(2,3)

            state = Statevector.from_instruction(qc).data

            features = []

            for i in range(4):

                p0, p1 = 0,0

                for idx, amp in enumerate(state):

                    if (idx >> (3-i)) & 1 == 0: p0 += abs(amp)**2

                    else: p1 += abs(amp)**2

                features.append(p0-p1)

            return np.array(features)

        h,w = img.shape

        feats = []

        for i in range(0, h, 8):

            for j in range(0, w, 8):

                patch = img[i:i+2, j:j+2]

                if patch.size < 4: continue

                feats.append(quantum_patch_feature(patch))

        return np.concatenate(feats)



    def method(self,img,name=None):

        if name is None:

            name=self.name

        match name:

            case "ALEXnet":

                return self.ALEXnet(img)

            case "HOG":

                return self.HOG(img)

            case "QFSD":

                return self.QFSD(img)

            case "quantum_inspired":

                return self.quantum_inspired(img)

            case _:

                return img

















class FaceDatasetBuilder:

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        self.fc = face_crop()



        self.rows = []

        self.failed = []

        self.total_files = 0

        self.total_failures = 0



    def build(self):

        for actor_name in os.listdir(self.root_dir):



            actor_path = os.path.join(self.root_dir, actor_name)



            if not os.path.isdir(actor_path):

                continue



            for img_name in os.listdir(actor_path):



                img_path = os.path.join(actor_path, img_name)



                if not os.path.isfile(img_path):

                    continue



                ext = os.path.splitext(img_name)[1].lower()



                if ext not in self.valid_exts:

                    continue



                self.total_files += 1



                try:

                    cropped = self.fc.face_cropped(

                        IMAGE_FILE=img_path,

                        padding=0.2,

                        output_size=(128, 128),

                        draw=False

                    )



                    if cropped is None:

                        self.total_failures += 1

                        self.failed.append((actor_name, img_name, "No face detected"))

                        print(f"ERROR | No face detected | {actor_name} | {img_name}")

                        continue



                    self.rows.append({

                        "actor_name": actor_name,

                        "image_name": img_name,

                        "image_path": img_path

                    })



                except Exception as e:

                    self.total_failures += 1

                    self.failed.append((actor_name, img_name, str(e)))

                    print(f"FAILED | {actor_name} | {img_name} | {e}")



        print(f"Total files: {self.total_files}")

        print(f"Total failures/errors: {self.total_failures}")

        print(f"Valid images added to dataset: {len(self.rows)}")



        df = pd.DataFrame(self.rows)

        return df







class FeatureDatasetBuilder:

    def __init__(

        self,

        face_df,

        selected_features=None,

        crop_size=(128, 128),

        padding=0.2

    ):

        self.face_df = face_df

        self.selected_features = selected_features or ["ALEXnet", "HOG", "QFSD", "quantum_inspired"]

        self.crop_size = crop_size

        self.padding = padding



        self.fc = face_crop()

        self.fe = feature_extraction()

        self.qip = QIP(name="QIIP")



        self.rows = []

        self.failed = []

        self.total_files = 0

        self.total_failures = 0



    def extract_one_image(self, img_path):

        cropped = self.fc.face_cropped(

            IMAGE_FILE=img_path,

            padding=self.padding,

            output_size=self.crop_size,

            draw=False

        )



        if cropped is None:

            raise ValueError("No face detected")



        fused_features = []



        for feature_name in self.selected_features:



            if feature_name == "ALEXnet":

                feat = self.fe.ALEXnet(cropped)



            elif feature_name == "HOG":

                feat = self.fe.HOG(cropped)



            elif feature_name == "QFSD":

                qiip_img = self.qip.QIIP(cropped)

                feat = self.fe.QFSD(qiip_img)



            elif feature_name == "quantum_inspired":

                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                feat = self.fe.quantum_inspired(gray)



            else:

                raise ValueError(f"Unknown feature: {feature_name}")



            feat = np.asarray(feat).flatten()

            fused_features.append(feat)



        return np.concatenate(fused_features)



    def build(self):

        for idx, row in self.face_df.iterrows():

            self.total_files += 1



            actor_name = row["actor_name"]

            img_name = row["image_name"]

            img_path = row["image_path"]



            try:

                fused = self.extract_one_image(img_path)



                new_row = {

                    "actor_name": actor_name,

                    "image_name": img_name,

                    "image_path": img_path

                }



                for i, value in enumerate(fused):

                    new_row[f"f_{i}"] = value



                self.rows.append(new_row)



            except Exception as e:

                self.total_failures += 1

                self.failed.append((actor_name, img_name, str(e)))

                print(f"FAILED | {actor_name} | {img_name} | {e}")



        print(f"Total files processed: {self.total_files}")

        print(f"Total failures/errors: {self.total_failures}")

        print(f"Valid feature rows: {len(self.rows)}")



        return pd.DataFrame(self.rows)







# Implements QIIP, QFSD, and quantum-inspired feature extraction.
