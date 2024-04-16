import numpy as np
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

# from perlin_numpy import generate_perlin_noise_2d


class Image2Augment:
    @staticmethod
    def apply_awgn(image, sigma=0.1):
        noisy_image = random_noise(image, mode="gaussian", var=sigma**2)
        return (255 * noisy_image).astype(np.uint8)

    # @staticmethod
    # def apply_apn(image):
    #     def map_to_nearest_power_of_two(n):
    #         if n <= 0:
    #             return 0
    #         power = 1
    #         while 2**power < n:
    #             power += 1
    #         return 2**power

    #     def add_noise_to_image(image, noise):
    #         resized_noise = cv2.resize(noise, (image.shape[1], image.shape[0]))
    #         noisy_image = np.clip(
    #             image.astype(np.float32) + resized_noise.astype(np.float32), 0, 255
    #         ).astype(np.uint8)

    #         return noisy_image

    #     size = max(image.shape)
    #     size = map_to_nearest_power_of_two(size)

    #     noise = generate_perlin_noise_2d((size, size), (8, 8))  # -- 2
    #     noisy_image = add_noise_to_image(image, noise)

    #     return noisy_image

    @staticmethod
    def apply_elastic_transform(image, alpha, sigma):
        # Elastic Transformations
        random_state = np.random.RandomState(None)
        height, width = image.shape[:2]
        dx = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # 좌표 배열을 생성하여 좌표에 변환을 적용
        distorted_x = np.clip(x + dx, 0, width - 1)
        distorted_y = np.clip(y + dy, 0, height - 1)

        print("--", distorted_x.shape)
        print("--", distorted_y.shape)

        distorted_image = map_coordinates(
            image,
            [distorted_y, distorted_x],
            order=1,
            mode="reflect",
        )
        distorted_image = distorted_image.reshape(image.shape)
        return distorted_image

    @staticmethod
    def apply_all_augmentations(
        image,
        et_small_alpha,
        et_small_sigma,
        et_large_alpha,
        et_large_sigma,
    ):
        image = Image2Augment.apply_awgn(image)
        # image = Image2Augment.apply_apn(image, apn_frequency, apn_max_intensity)
        image = Image2Augment.apply_elastic_transform(
            image, et_small_alpha, et_small_sigma
        )
        image = Image2Augment.apply_elastic_transform(
            image, et_large_alpha, et_large_sigma
        )
        return image
