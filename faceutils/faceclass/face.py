# Python Basic Libraries
from __future__ import annotations
from typing import List, NoReturn, Tuple, Union, Any
from os.path import basename, dirname, realpath, join as path_join, exists as path_exists
import os
from uuid import uuid4
from copy import copy
import cv2

# Python Libraries
from cv2 import imread, cvtColor, COLOR_BGR2RGB, resize, imwrite, imshow,\
    waitKey, circle, putText, FONT_HERSHEY_SIMPLEX, addWeighted, fillPoly
import dlib
import mediapipe as mp
from numpy import array as np_array, int32, zeros as np_zeros, save as np_save,\
    max as np_max, uint8, concatenate as np_concatenate, mean as np_mean,\
    where as np_where, min as np_min, std as np_std, full as np_full
import imutils
from faceutils.facealign import align


# Modules
from faceutils.resnet_mask import FaceParser


#==============================================================================
# Constants
#==============================================================================
# Landmarks constants
LANDMARKS_DLIB = 0
LANDMARKS_MEDIAPIPE = 1
LANDMARKS_FACE2POINTS = 2
LANDMARKS_DLIB_FACE2POINTS = 3
LANDMARKS_MEDIAPIPE_FACE2POINTS = 4


# Path to the semantic folder as default
SEMANTIC_FOLDER_NAME: str = "semantic"
LANDMARKS_FOLDER_NAME: str = "landmarks"

USED_COLOR_OVERLAY: np_array = np_array([
    [255, 35, 35],
    [187, 210, 14],
    [128, 102, 164],
    [163, 192, 204],
    [56, 72, 35],
    [79, 58, 43],
    [113, 75, 84],
    [180, 146, 132],
    [245, 16, 44],
    [223, 138, 174],
    [13, 120, 124],
    [153, 202, 233],
    [6, 143, 196],
    [6, 196, 181],
    [6, 196, 98],
    [33, 5, 196],
    [134, 6, 196],
    [196, 6, 147],
    [196, 6, 6],
    [54, 248, 3],
    [242, 167, 5]
])


#==============================================================================
# Module initialisation 
#==============================================================================
# Face parser used to determine the semantic of the face
face_parser = FaceParser()


# Get the dlib model
DLIB_DATA_DIR: str = os.environ.get(
    'DLIB_DATA_DIR',
    path_join(dirname(dirname(realpath(__file__))), 'dlib')
)
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor(
    path_join(DLIB_DATA_DIR, 'shape_predictor_68_face_landmarks.dat')
)

# Get the face2points model
FACE_2_POINTS_DATA_DIR: str = path_join(
    dirname(dirname(realpath(__file__))), 'facepoints'
)
face2points_predictor = dlib.shape_predictor(
    path_join(FACE_2_POINTS_DATA_DIR, 'scaleall500_withbox_depth3_oversample60_casdep15_featpool1500.dat')
)


#==============================================================================
# Class
#==============================================================================
class Face:
    """
    Tool box for face on image
    """

    # Class attributes
    def __init__(
        self, path: str, image: np_array = None, semantic: np_array = None
    ) -> NoReturn:
        """
        Constructor

        :param path: Path to the image
        :type path: str
        :param image: Image (in BGR format), defaults to None
        :type image: np_array, optional
        :param semantic: Semantic of the image, defaults to None
        :type semantic: np_array, optional
        :return: Nothing it's a constructor ;)
        :rtype: NoReturn
        """
        self.path: str = path
        self.image: np_array = image
        self.semantic: np_array = semantic

    def read_image(self, path=None):
        if path is None:
            path = self.path

        self.image = imread(path)


    def build(self, semantic_path: str = None) -> NoReturn:
        """
        Read the image and the semantic based on the path

        :param semantic_path: Path to the semantic data, defaults to None
        :type semantic_path: str, optional
        :return: Nothing
        :rtype: NoReturn
        """
        # If the semantic has not been precised
        if semantic_path is None:
            # Determine the semantic folder based on the image folder
            semantic_path: str = path_join(dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME, basename(self.path))

        # Read the image and the semantic
        self.image = imread(self.path)
        self.semantic = imread(self.semantic_path)

        # Assert they have the same size
        if self.image.shape[:2] != self.semantic.shape[:2]:
            resize(self.semantic, self.image.shape[:2])
            
    def landmark(self, landmark_model: int = LANDMARKS_MEDIAPIPE) -> np_array:
        """
        Compute the landmark of the face

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Landmarks of the face
        :rtype: np_array
        """

        # For dlib landmarks
        if landmark_model == LANDMARKS_DLIB:
            # Convert the image to rgb
            rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
            rects: np_array = dlib_detector(rgbimg, 1)

            # If no face found raise an error
            if len(rects) == 0:
                raise RuntimeError("No face found !")
                return self.landmark()

            # Determine the landmarks
            shapes = dlib_predictor(rgbimg, rects[0])
            # Parse the points
            points: np_array = np_array([(shapes.part(i).x, shapes.part(i).y) for i in range(68)], int32)
        elif landmark_model == LANDMARKS_FACE2POINTS:
            # Convert the image to rgb
            rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
            rects: np_array = dlib_detector(rgbimg, 1)

            # If no face found raise an error
            if len(rects) == 0:
                raise RuntimeError("No face found !")
                return self.landmark()

            # Determine the landmarks
            shapes: np_array = face2points_predictor(rgbimg, rects[0])
            # Parse the points
            points: np_array = np_array([(
                shapes.part(i).x, shapes.part(i).y) for i in range(shapes.shape[0])
            ], int32)
        # For mediapipe landmarks
        elif landmark_model == LANDMARKS_MEDIAPIPE:
            # Get the size of the image (it will be usefull for the landmarks points since they are normalized)
            width, height = self.size
            # Create the model
            face_mesh: mp.solutions.Solution = mp.solutions.face_mesh.FaceMesh(
		        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
	        )
            # Get the mesh of the face
            mesh: mp.solutions.Solution = face_mesh.process(cvtColor(self.image, COLOR_BGR2RGB))
            # Parse the points
            points: np_array = np_array([
                (lm.x * height, lm.y * width) for lm in mesh.multi_face_landmarks[0].landmark
            ])
        elif landmark_model == LANDMARKS_DLIB_FACE2POINTS:
            # Convert the image to rgb
            rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
            rects: np_array = dlib_detector(rgbimg, 1)

            # If no face found raise an error
            if len(rects) == 0:
                raise RuntimeError("No face found !")
                return self.landmark()

            # Determine the landmarks
            dlib_shapes = dlib_predictor(rgbimg, rects[0])
            face2points_shapes: np_array = face2points_predictor(rgbimg, rects[0])
            # Parse the points
            points_dlib: np_array = np_array([
                (dlib_shapes.part(i).x, dlib_shapes.part(i).y) for i in range(68)
            ], int32)
            points_face2points: np_array = np_array([
                (face2points_shapes.part(i).x, face2points_shapes.part(i).y) for i in range(127, 138)
            ], int32)
            # Concats the parsed points
            points = np_concatenate((points_dlib, points_face2points), axis=0)
        elif landmark_model == LANDMARKS_MEDIAPIPE_FACE2POINTS:
            # Convert the image to rgb
            rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
            rects: np_array = dlib_detector(rgbimg, 1)

            # If no face found raise an error
            if len(rects) == 0:
                raise RuntimeError("No face found !")
                return self.landmark()

            # Determine the landmarks
            face2points_shapes: np_array = face2points_predictor(rgbimg, rects[0])
            # Parse the points
            # Get the size of the image (it will be usefull for the landmarks points since they are normalized)
            width, height = self.size
            # Create the model
            face_mesh: mp.solutions.Solution = mp.solutions.face_mesh.FaceMesh(
		        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
	        )
            # Get the mesh of the face
            mesh: mp.solutions.Solution = face_mesh.process(cvtColor(self.image, COLOR_BGR2RGB))
            # Parse the points
            points_mediapipe: np_array = np_array([
                (lm.x * height, lm.y * width) for lm in mesh.multi_face_landmarks[0].landmark
            ])
            points_face2points: np_array = np_array([
                (face2points_shapes.part(i).x, face2points_shapes.part(i).y) for i in range(127, 138)
            ], int32)
            # Concats the parsed points
            points = np_concatenate((points_mediapipe, points_face2points), axis=0)

        return points

    def face_crop(
        self, landmark_model: int = LANDMARKS_MEDIAPIPE, auto_update: bool = True
    ) -> Tuple[np_array, np_array]:
        """
        Crop over a single face

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param auto_update: True to update the class, False either, defaults to True
        :type auto_update: bool, optional
        :return: Crop over a face in the image and in the semantic
        :rtype: Tuple[np_array, np_array]
        """

        width, heigh = self.size
        # Get the landmark
        landmarks: np_array = self.landmark(landmark_model)

        # Transpose the array to have seperatly the x in array and the y in an other one
        landmarks_T: np_array = landmarks.T

        # Determine the limits of landmarks
        max_x, max_y = max(landmarks_T[0]), max(landmarks_T[1])
        min_x, min_y = min(landmarks_T[0]), min(landmarks_T[1])

        # Build a rectantangle over the landmarks and crop over
        crop_image: np_array = self.image[
            max(0, min_y * 0.9):min(width, max_y * 1.1), max(0, min_x * 0.9):min(heigh, max_x * 1.25)
        ]
        crop_semantic: np_array = self.image[
            max(0, min_y * 0.9):min(width, max_y * 1.1), max(0, min_x * 0.9):min(heigh, max_x * 1.25)
        ]

        # If the autoupdate is True, update the value of its own attributes
        if auto_update:
            self.image = crop_image
            self.semantic = crop_semantic

        return crop_image, crop_semantic

    def square_crop(
        self, landmark_model: int = LANDMARKS_DLIB_FACE2POINTS, auto_update: bool = True
    ) -> Tuple[np_array, np_array]:
        """
        Create a crop arround a single face in a square shape

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param auto_update: True to update the class, False otherwise, defaults to True
        :type auto_update: bool, optional
        :raises NotImplementedError: An impossible case
        :return: Crop over a face in the image and in the semantic
        :rtype: Tuple[np_array, np_array]
        """

        # Get the size of the image
        width, heigh = self.size
        # Get the landmark
        landmarks: np_array = self.landmark(landmark_model)

        # Transpose the array to have seperatly the x in array and the y in an other one
        landmarks_T: np_array = landmarks.T

        # Determine the limits of landmarks
        max_x = min(heigh, max(landmarks_T[0]) + ((max(landmarks_T[0]) - min(landmarks_T[0])) * 0.05)) 
        max_y = min(width, max(landmarks_T[1]) + ((max(landmarks_T[1]) - min(landmarks_T[1])) * 0.1))
        min_x = max(0, min(landmarks_T[0]) - ((max(landmarks_T[0]) - min(landmarks_T[0])) * 0.05))
        min_y = max(0, min(landmarks_T[1]) - ((max(landmarks_T[1]) - min(landmarks_T[1])) * 0.1))

        # Compute the size of the crop
        nb_px_x: float = max_x - min_x
        nb_px_y: float = max_y - min_y

        # (y, width)
        # ^
        # |
        # .___> (x, heigh)
        #
        if nb_px_x == nb_px_y:
            pass
        elif nb_px_x > nb_px_y and nb_px_x <= width:
            # Compute the adding to min_x and max_x
            decrease_min_y: int = min((nb_px_x - nb_px_y) // 2, min_y)
            increase_max_y: int = min((nb_px_x - nb_px_y) - decrease_min_y, width - max_y)
            # Add the rest to min_x
            decrease_min_y += ((nb_px_x - nb_px_y) - decrease_min_y) - increase_max_y
            # Update the value y
            min_y, max_y = min_y - decrease_min_y, max_y + increase_max_y
        elif nb_px_y > nb_px_x and nb_px_y <= heigh:
            # Compute the adding to min_y and max_y
            decrease_min_x: int = min((nb_px_y - nb_px_x) // 2, min_x)
            increase_max_x: int = min((nb_px_y - nb_px_x) - decrease_min_x, heigh - max_x)
            # Add the rest to min_y
            decrease_min_x += ((nb_px_y - nb_px_x) - decrease_min_x) - increase_max_x
            # Update the value x
            min_x, max_x = min_x - decrease_min_x, max_x + increase_max_x
        elif nb_px_x > nb_px_y:
            # Increase at the maximum possible to loose the min of information
            min_y, max_y = 0, width
            # Update the nb_px_y
            nb_px_y = width

            # Compute the adding to min_y and max_y
            increase_min_x: int = min((nb_px_x - nb_px_y) // 2, min_x)
            decrease_max_x: int = min((nb_px_x - nb_px_y) - increase_min_x, heigh - max_x)
            # Add the rest to min_y
            increase_min_x += ((nb_px_x - nb_px_y) - increase_min_x) - decrease_max_x
            # Update the value x
            min_x, max_x = min_x + increase_min_x, max_x - decrease_max_x
        elif nb_px_y > nb_px_x:
            # Increase at the maximum possible to loose the min of information
            min_x, max_x = 0, heigh
            # Update the nb_px_y
            nb_px_x = heigh

            # Compute the adding to min_y and max_y
            increase_min_y: int = min((nb_px_y - nb_px_x) // 2, min_y)
            decrease_max_y: int = min((nb_px_y - nb_px_x) - increase_min_y, width - max_y)
            # Add the rest to min_y
            increase_min_y += ((nb_px_y - nb_px_x) - increase_min_y) - decrease_max_y
            # Update the value x
            min_y, max_y = min_y + increase_min_y, max_y - decrease_max_y
        else:
            raise NotImplementedError("Hey where are we ? Is it even possible ?") # ;)

        # Build a rectantangle over the landmarks and crop over
        crop_image: np_array = self.image[int(min_y):int(max_y), int(min_x):int(max_x)]
        crop_semantic: np_array = self.semantic[int(min_y):int(max_y), int(min_x):int(max_x)]

        # If the autoupdate is True, update the value of its own attributes
        if auto_update:
            self.image = crop_image
            self.semantic = crop_semantic

        return crop_image, crop_semantic

    @property
    def name(self) -> str:
        """
        Name getter

        :return: The name of the image
        :rtype: str
        """
        return basename(self.path).split(".")[0]

    @property
    def size(self) -> Tuple[int, int]:
        """
        Size getter

        :return: The shape of the image
        :rtype: Tuple[int, int]
        """
        return self.image.shape[:2]

    def resize(self, size: Tuple[int, int]) -> NoReturn:
        """
        Resize the image and the semantic

        :param size: New size of image and semantic
        :type size: Tuple[int, int]
        :return: Nothing it just update itself :p
        :rtype: NoReturn
        """
        self.image = resize(self.image, size)
        try:
            self.semantic = resize(self.semantic, size)
        except cv2.error:
            pass

    def save(
        self, semantic_path: str = None, landmarks_path: str = None,
        overwrite: bool = True, save_landmarks: bool = False,
        save_semantic: bool = False,
        landmark_model: int = LANDMARKS_MEDIAPIPE
    ) -> NoReturn:
        """
        Save the image, semantic and landmarks

        :param semantic_path: Path to the semantic file, defaults to None
        :type semantic_path: str, optional
        :param landmarks_path: Path to the landmark file, defaults to None
        :type landmarks_path: str, optional
        :param overwrite: True if it can overwrite the previous file, else False, defaults to True
        :type overwrite: bool, optional
        :param save_landmarks: True if the landmarks have to be saved too, False otherwise, defaults to True
        :type save_landmarks: bool, optional
        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Nothing
        :rtype: NoReturn
        """

        # If the save can overwrite the images
        if overwrite:

            # If the semantic is not define
            if semantic_path is None:
                # Determine the semantic folder based on the image folder
                semantic_path: str = path_join(
                    dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME,
                    basename(self.path)
                )

            # If the landmarks path is not define
            if landmarks_path is None:
                # Determine the landmarks folder based on the image folder
                landmarks_path: str = path_join(
                    dirname(dirname(self.path)), LANDMARKS_FOLDER_NAME,
                    basename(self.path).split(".")[0] + ".npy"
                )

        # If the semantic has not been precised
        elif not overwrite:
            # Create an unique name
            img_name: str = str(uuid4())
            # Update the path
            self.path = path_join(dirname(self.path), f"{img_name}.png")
            # Determine the semantic folder based on the image folder
            semantic_path: str = path_join(
                dirname(dirname(self.path)), SEMANTIC_FOLDER_NAME, f"{img_name}.png"
            )
            # Determine the landmarks folder based on the image folder
            landmarks_path: str = path_join(
                dirname(dirname(self.path)), LANDMARKS_FOLDER_NAME, f"{img_name}.npy"
            )

        # Verify that the path exists
        # If the image folder does not exists create it
        if not path_exists(dirname(self.path)):
            os.mkdir(dirname(self.path))

        # If the semantic folder does not exists create it
        if not path_exists(dirname(semantic_path)):
            os.mkdir(dirname(semantic_path))

        # If the landmarks folder does not exists create it
        if not path_exists(dirname(landmarks_path)):
            os.mkdir(dirname(landmarks_path))

        # Save the image
        imwrite(self.path, self.image)
        # Save the semantic
        if save_semantic:
            imwrite(semantic_path, self.semantic)

        # If the landmarks have to be saved
        if save_landmarks:
            # Get the landmarks
            landmarks = self.landmark(landmark_model=landmark_model)
            # Save the landmarks
            np_save(landmarks_path, np_array(landmarks))

    def process(
        self, landmark_model: int = LANDMARKS_MEDIAPIPE,
        size: Tuple[int, int] = (512, 512)
    ) -> NoReturn:
        """
        Create the semantic and a square crop of the image

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :param size: Size of the image, defaults to (512, 512)
        :type size: Tuple[int, int], optional
        :raises UserWarning: If the size is not square
        :return: Nothing but do a lot of thing behind ;)
        :rtype: NoReturn
        """

        # The processing has to work on non empty images
        assert self.image is not None or self.path is not None

        if self.image is None:
            self.image: np_array = imread(self.path)

        # Set the semantic to 0
        self.semantic = np_zeros(shape=self.image.shape, dtype=int32)
        # Crop the image into a square to avoid modifying the shape of the image
        self.square_crop(landmark_model)

        # Resize the image to fit the input size of the network which will parse the image
        self.image = resize(self.image, (512, 512))

        # Get the semantic of the image
        self.semantic = face_parser.parse(image=self.image).numpy()
        self.semantic = self.semantic.astype(uint8)

        # Resize to get the wanted shape
        if size[0] != size[1]:
            raise UserWarning(
                "The size entered is not a square size, which will modify the global shape of the image."
            )

        self.resize(size)
        return self

    @classmethod
    def empty(cls, path: str = None) -> Face:
        """
        Create an empty Face object

        :param path: Path to the image file, defaults to None
        :type path: str, optional
        :return: Itself
        :rtype: Face
        """
        if path is None:
            # Create a new path
            path: str = path_join(
                dirname(realpath(__file__)), "images", f"{str(uuid4())}.png"
            )
        # Return a new instance
        return cls(path=path)

    @classmethod
    def from_image(cls, image: np_array) -> Face:
        """
        Create an instance of the class from an image

        :param image: Image to create the object on
        :type image: np_array
        :return: Itself ;)
        :rtype: Face
        """
        # Crate an empty Face Object
        empty = cls.empty()
        # Add the image to it
        empty.image = image
        # Return the empty Face object with the image
        return empty

    def show(self) -> NoReturn:
        """
        Show the image

        :return: _description_
        :rtype: NoReturn
        """
        imshow(self.path, self.image)
        waitKey(0)

    def get_mask(self, semantic_value: Union[int, list]) -> np_array:
        """
        Show a specified element in the image

        :param semantic_value: The value of the semantic we are searching for
        :type semantic_value: int
        :return: The element wanted
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Every value that is not in the mask will be black
        return mask

    def get_element(self, semantic_value: Union[int, list]) -> np_array:
        """
        Show a specified element in the image

        :param semantic_value: The value of the semantic we are searching for
        :type semantic_value: int
        :return: The element wanted
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get a 3d mask for every color
        mask_colour: np_array = np_array([mask.T, mask.T, mask.T])

        # Every value that is not in the mask will be black
        return self.image * mask_colour.T

    def crop_over_element(self, semantic_value: Union[int, list]) -> np_array:
        """
        Crop over the wanted semantic element

        :param semantic_value: _description_
        :type semantic_value: Union[int, list]
        :return: _description_
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get the indices that are in the mask
        indices: np_array = np_where(mask > 0)
        crop_min_x, crop_max_x = np_min(indices[0]), np_max(indices[0])
        crop_min_y, crop_max_y = np_min(indices[1]), np_max(indices[1])
        # Get a 3d mask for every color
        mask_colour: np_array = np_array([mask.T, mask.T, mask.T])
        # Get the masked image
        maked_image: np_array = self.image * mask_colour.T
        # Get the image cropped
        cropped_image: np_array = maked_image[crop_min_x:crop_max_x, crop_min_y:crop_max_y, :]

        return cropped_image

    def crop_over_mask(self, semantic_value: Union[int, list]) -> np_array:
        """
        Crop over the wanted semantic element

        :param semantic_value: _description_
        :type semantic_value: Union[int, list]
        :return: _description_
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Get the indices that are in the mask
        indices: np_array = np_where(mask > 0)
        crop_min_x, crop_max_x = np_min(indices[0]), np_max(indices[0])
        crop_min_y, crop_max_y = np_min(indices[1]), np_max(indices[1])
        # Get the image cropped
        cropped_mask: np_array = mask[crop_min_x:crop_max_x, crop_min_y:crop_max_y]

        return cropped_mask


    def show_element(
        self, semantic_value: Union[int, list], with_crop: bool = False
    ) -> NoReturn:
        """
        Show a semantic element in the image

        :param semantic_value:  The value of the semantic we are searching for
        :type semantic_value: Union[int, list]
        :return: No return
        :rtype: NoReturn
        """
        # Show the semantic element
        if not with_crop:
            image_semantic: np_array = self.get_element(semantic_value)
        else:
            image_semantic: np_array = self.crop_over_element(semantic_value)

        # Show the element
        imshow("Semantic value", image_semantic)
        waitKey()


    def show_semantic(self, alpha: float = 0.4) -> NoReturn:
        """
        Show the mask over the image

        :param alpha: Transparency of the overlay between 0 and 1, defaults to 0.4
        :type alpha: float, optional
        :return: Nothing
        :rtype: NoReturn
        """
        colour_overlay: np_array = np_zeros((*self.size[:2], 3), dtype=uint8)
        colour_overlay[:, :, 0] += self.semantic

        for colour in range(int(np_max(self.semantic))):
            # Get the masks
            mask = colour_overlay[:, :, 0] == colour

            # Change the color
            colour_overlay[mask] = USED_COLOR_OVERLAY[colour]

        imshow("Mask", addWeighted(colour_overlay, alpha, self.image, 1 - alpha, 0))
        waitKey()

    def show_landmarks(self, landmark_model: int = LANDMARKS_MEDIAPIPE) -> NoReturn:
        """
        Show the landmarks numeroted on the image

        :param landmark_model: Model to use to compute the landmarks, {0, 1, 2}, defaults to 0
        :type landmark_model: int, optional
        :return: Nothing
        :rtype: NoReturn
        """
        # Get the landmark
        landmarks: np_array = self.landmark(landmark_model)

        # Copy the image
        img_copy = copy(self.image)
        # Write image + all the found cordinate points (x,y)
        for k, (x, y) in enumerate(landmarks):
            # write points
            img_copy = circle(img_copy, (int(x), int(y)), 2, (0, 160, 255), -1)
            # write numbers
            img_copy = putText(
                img_copy, text=str(k), org =(int(x)+2, int(y)+2),
                color=(0, 160, 255), fontFace=FONT_HERSHEY_SIMPLEX,
                fontScale=0.3
            )

        img_copy = imutils.resize(img_copy, width=500)
        imshow(self.path, img_copy)
        waitKey(0)

    def __getitem__(self, items: Any) -> Union[str, np_array]:
        """
        Item getter

        :param items: Items index
        :type items: Any
        :return: Items
        :rtype: Union[str, np_array]
        """
        elements: list = [self.path, self.image, self.semantic]
        return elements[items]

    def align(
        self, other_face: Union[List[Face], Face], landmark_model: int = LANDMARKS_MEDIAPIPE,
        self_update: bool = False
    ) -> Face:

        # If the other face is a single face
        if isinstance(other_face, Face):
            # set it to a list
            other_face = [other_face]

        # Get the faces and landmarks
        faces = [self.image] + [face.image for face in other_face]
        landmarks = [self.landmark(landmark_model)] + [face.landmark(landmark_model) for face in other_face]

        # Align the faces
        # Skip the first one since it's the reference face, so its the same as the original
        aligned_faces: List[np_array] = align(images=faces, all_points=landmarks, landmark_model=landmark_model)[1:]

        # If we actualy need to update the faces object
        if self_update:
            # Update every faces
            for face, aligned in zip(other_face, aligned_faces):
                # Update the image
                face.image = aligned
                face.process(landmark_model, face.size)

        return aligned_faces


    @property
    def has_face(self) -> bool:
        """
        Return True if face is found on the image

        :return: True if face is found on the image else False
        :rtype: bool
        """

        # Convert the image to rgb
        rgbimg: np_array = cvtColor(self.image, COLOR_BGR2RGB)
        rects: np_array = dlib_detector(rgbimg, 1)
        # Return if the number of faces found is above 0
        return len(rects) > 0

    def __copy__(self) -> Face:
        return Face(self.path, self.image, self.semantic)

    def get_zone_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 0] = 0

        return image_copy

    def get_mask_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to keep
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        return mask

    def exclude_zone_landmarks(
        self,  landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param landmarks_points: List of points to define the area to remove
        :type landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[landmarks_points]], color=1)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 1] = 0

        return image_copy


    def segmentation_zone_landmarks(
        self,  keep_landmarks_points: List,
        exlude_landmarks_points: List, landmark_model: int = LANDMARKS_DLIB
    ) -> np_array:
        """
        Get a given zone defined by landmarks points

        :param keep_landmarks_points: List of points to define the area to keep
        :type keep_landmarks_points: List
        :param exlude_landmarks_points: List of points to define the area to keep
        :type exlude_landmarks_points: List
        :param landmark_model: Model to use for landmarks, defaults to 0
        :type landmark_model: int, optional
        :return: A masked version of the original image
        :rtype: np_array
        """
        # Get the landmarks
        landmarks: np_array = self.landmark(landmark_model=landmark_model)
        # Create a mask
        mask: np_array = np_zeros(self.size)
        # Create the polly to keep
        mask = fillPoly(mask, pts=[landmarks[keep_landmarks_points]], color=1)
        # Create the polly to remove
        mask = fillPoly(mask, pts=[landmarks[exlude_landmarks_points]], color=0)

        # Apply the mask
        # First copy the image
        image_copy: np_array = copy(self.image)
        # Apply the mask
        image_copy[mask == 0] = 0

        return image_copy

    def get_mask(self, semantic_value: Union[int, list]) -> np_array:
        """
        Show a specified element in the image

        :param semantic_value: The value of the semantic we are searching for
        :type semantic_value: int
        :return: The element wanted
        :rtype: np_array
        """

        # Check the instance of the semantic value given
        if isinstance(semantic_value, list):
            # Get a mask for every value given
            mask = self.semantic == semantic_value[0]
            for value in semantic_value[1:]:
                mask += self.semantic == value

            mask = mask > 0
        elif isinstance(semantic_value, float) or isinstance(semantic_value, int):
            mask = self.semantic == semantic_value
        else:
            raise ValueError(f"The type {type(semantic_value)} is unsupported for the parameter semantic_value.")

        # Every value that is not in the mask will be black
        return mask.astype("uint8")