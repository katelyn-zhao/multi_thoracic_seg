import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from scipy.spatial.distance import cdist
from scipy.ndimage import rotate


raw_image_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/raw_images/'
mask_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/segmentations/'
predictions_directory = 'C:/Users/Mittal/Desktop/thoracic_seg/unet_niipredictions/'

raw_images = sorted(os.listdir(raw_image_directory))
segmentation_images = sorted(os.listdir(mask_directory))
predictions = sorted(os.listdir(predictions_directory))

unet_prediction_dataset = []
raw_image_dataset = []
segmentation_dataset = []
image_names = []

for image_name in predictions:    
    if (image_name.split('.')[1] == 'nii'):
        base_name = image_name.split('.')[0]
        image = nib.load(predictions_directory+image_name).get_fdata()
        raw = nib.load(raw_image_directory+image_name).get_fdata()
        mask = nib.load(mask_directory+image_name).get_fdata()
        unet_prediction_dataset.append(np.array(image))
        raw_image_dataset.append(np.array(raw))
        segmentation_dataset.append(np.array(mask))
        image_names.append(image_name)

def dice_coef_p(y_true, y_pred, smooth=1.):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


def affine_registration_3d(fixed_image, moving_image, 
                         initial_transform=None,
                         optimizer='gradient_descent',
                         metric='mutual_information',
                         learning_rate=1.0,
                         min_step=1e-4,
                         iterations=200,
                         shrink_factors=[4, 2, 1],
                         smoothing_sigmas=[2, 1, 0],
                         sampling_percentage=0.25,
                         lock_rotation=True,
                         verbose=False):
    """
    Perform affine registration between two 3D MRI images with proper type handling.
    """
    
    # Ensure images are in float format (required for registration)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # Initialize registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Set similarity metric
    if metric.lower() == 'mutual_information':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric.lower() == 'mean_squares':
        registration_method.SetMetricAsMeanSquares()
    elif metric.lower() == 'correlation':
        registration_method.SetMetricAsCorrelation()
    else:
        raise ValueError("Unsupported metric. Choose 'mutual_information', 'mean_squares', or 'correlation'")
    
    registration_method.SetMetricSamplingPercentage(sampling_percentage, sitk.sitkWallClock)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    
    # Set optimizer
    if optimizer.lower() == 'gradient_descent':
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=learning_rate,
            minStep=min_step,
            numberOfIterations=iterations,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-4)
    elif optimizer.lower() == 'lbfgsb':
        registration_method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=iterations,
            maximumNumberOfCorrections=5)
    elif optimizer.lower() == 'exhaustive':
        registration_method.SetOptimizerAsExhaustive(numberOfSteps=[10, 10, 10, 5, 5, 5])
    else:
        raise ValueError("Unsupported optimizer. Choose 'gradient_descent', 'lbfgsb', or 'exhaustive'")
    
    # Set multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Initialize transform
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.AffineTransform(fixed_image.GetDimension()),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # if lock_rotation:
    #     registration_method.SetOptimizerScalesFromPhysicalShift()
        
    #     # For AffineTransform, parameters are ordered as:
    #     # [m11, m12, m13, m21, m22, m23, m31, m32, m33, tx, ty, tz]
    #     optimizer_scales = [0.01, 0.01, 0.01,   # First row (m11, m12, m13)
    #                0.01, 0.01, 0.01,     # Second row (m21, m22, m23)
    #                0.01, 0.01, 0.01,     # Third row (m31, m32, m33)
    #                0.01, 0.01, 0.01]              # Translation (tx, ty, tz)
        
    #     registration_method.SetOptimizerScales(optimizer_scales)
    
    # Set interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Execute registration
    if verbose:
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: print(
            f"Iteration: {registration_method.GetOptimizerIteration()}, "
            f"Metric value: {registration_method.GetMetricValue():.4f}"))
    
    try:
        final_transform = registration_method.Execute(fixed_image, moving_image)
    except RuntimeError as e:
        print(f"Registration failed: {str(e)}")
        # Return identity transform if registration fails
        final_transform = sitk.AffineTransform(fixed_image.GetDimension())
    
    # Apply the transform to the moving image
    transformed_moving_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID())
    
    # Get final metric value
    registration_metric_value = registration_method.GetMetricValue()
    
    if verbose:
        print(f"Final metric value: {registration_metric_value:.4f}")
        print(f"Optimizer stopping condition: {registration_method.GetOptimizerStopConditionDescription()}")
        print(f"Transform parameters: {final_transform.GetParameters()}")
    
    return transformed_moving_image, final_transform, registration_metric_value

def preprocess_image(image_array):
    """Apply your preprocessing steps to the numpy array"""
    # Example: Create binary mask where segmentation value == 1
    processed_array = np.where(image_array == 1, 1, 0)
    return processed_array

def affine_registration_with_preprocessing(fixed_image_path, moving_image_path, 
                                         fixed_seg_path=None, moving_seg_path=None):
    """
    Perform affine registration with optional preprocessing based on segmentations.
    
    Parameters:
    - fixed_image_path: Path to fixed image (target)
    - moving_image_path: Path to moving image (to be registered)
    - fixed_seg_path: Optional path to fixed image segmentation
    - moving_seg_path: Optional path to moving image segmentation
    """
    
    # Load original images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    
    # If segmentation masks are provided, preprocess the images
    if fixed_seg_path and moving_seg_path:
        # Load segmentations
        fixed_seg = sitk.ReadImage(fixed_seg_path)
        moving_seg = sitk.ReadImage(moving_seg_path)
        
        # Convert to numpy arrays
        fixed_seg_array = sitk.GetArrayFromImage(fixed_seg)
        moving_seg_array = sitk.GetArrayFromImage(moving_seg)
        
        # Apply your preprocessing (e.g., create binary masks)
        fixed_mask_array = preprocess_image(fixed_seg_array)
        moving_mask_array = preprocess_image(moving_seg_array)
        
        # Convert back to SimpleITK images
        fixed_mask = sitk.GetImageFromArray(fixed_mask_array)
        moving_mask = sitk.GetImageFromArray(moving_mask_array)
        
        # Copy metadata from original images
        fixed_mask.CopyInformation(fixed_image)
        moving_mask.CopyInformation(moving_image)
        
        # Use the masks for registration instead of original images
        fixed_image = fixed_mask
        moving_image = moving_mask
    
    # Perform registration (using the function from previous example)
    registered_image, transform, metric_value = affine_registration_3d(
        fixed_image, 
        moving_image,
        verbose=True
    )
    
    return registered_image, transform, metric_value

def apply_transform(input_image, transform, reference_image=None, interpolator=sitk.sitkLinear, default_value=0.0):
    """
    Apply a transform to an image.
    
    Parameters:
    - input_image: The image to transform
    - transform: The transform to apply
    - reference_image: The image defining the output space (optional)
    - interpolator: Interpolation method (sitk.sitkLinear, sitk.sitkNearestNeighbor, etc.)
    - default_value: Value used for pixels outside the transformed image
    
    Returns:
    - The transformed image
    """
    # If no reference image provided, use the input image's space
    if reference_image is None:
        reference_image = input_image
    
    # Apply the transform
    transformed_image = sitk.Resample(
        input_image,
        reference_image,
        transform,
        interpolator,
        default_value,
        input_image.GetPixelID())
    
    return transformed_image

def pad_volume_z(volume):
    pad_z = max(0, 64 - volume.shape[2])  # Pad the z-direction
    pad_z_begin = pad_z // 2
    pad_z_end = pad_z - pad_z_begin
    pad_width = ((0, 0), (0, 0), (pad_z_begin, pad_z_end))  # Padding only in z-direction
    volume_padded = np.pad(volume, pad_width, mode='constant', constant_values=0.0)
    return volume_padded

print('Calculating Distances...')

distances = []

for i in range(len(raw_image_dataset) - 1):
    test = pad_volume_z(unet_prediction_dataset[-1])

    comparison = pad_volume_z(segmentation_dataset[i])
    comparison = np.where(comparison == 1, 1, 0)

    dist = dice_coef_p(test, comparison)
    distances.append(dist)

closest = np.argmax(distances)

plt.figure(figsize=(16, 8))
plt.subplot(141)
plt.title(f'{image_names[-1]}')
plt.imshow(unet_prediction_dataset[-1][:,:,unet_prediction_dataset[-1].shape[2]//2], cmap='gray')
plt.subplot(142)
plt.title(f'{image_names[closest]}')
plt.imshow(segmentation_dataset[closest][:,:,segmentation_dataset[closest].shape[2]//2], cmap='gray')
plt.show()
plt.close()

print('Finished calculating distances.')

print('Starting Registration...')

# Paths to your images and segmentations
fixed_img_path = predictions_directory+predictions[-1]
moving_img_path = mask_directory+segmentation_images[closest]
fixed_seg_path = predictions_directory+predictions[-1] # Optional
moving_seg_path = mask_directory+segmentation_images[closest]  # Optional

# Perform registration with preprocessing
registered_img, transform, metric = affine_registration_with_preprocessing(
    fixed_image_path=fixed_img_path,
    moving_image_path=moving_img_path,
    fixed_seg_path=fixed_seg_path,
    moving_seg_path=moving_seg_path
)

# Save the result
sitk.WriteImage(registered_img, f'C:/Users/Mittal/Desktop/thoracic_seg/test_registered.nii')
sitk.WriteTransform(transform, f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.tfm')

# affine_registration(pad_volume_z(np.where(segmentation_dataset[0] == 1, 1, 0)), 
#                     pad_volume_z(np.where(segmentation_dataset[closest+1] == 1, 1, 0)), 
#                     f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.txt',
#                     f'C:/Users/Mittal/Desktop/thoracic_seg/test_registered.nii')

print('Finished Registration')

loaded_transform = sitk.ReadTransform(f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.tfm')
new_image = sitk.ReadImage(mask_directory+segmentation_images[closest])
fixed_image = sitk.ReadImage(predictions_directory+predictions[-1])


transformed_image = apply_transform(new_image, loaded_transform, fixed_image)
sitk.WriteImage(transformed_image, f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.nii')


# apply_transformation(pad_volume_z(segmentation_dataset[0]),
#                      pad_volume_z(segmentation_dataset[closest+1]),
#                      f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.txt',
#                      f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.nii')

new_prediction = np.array(nib.load(f'C:/Users/Mittal/Desktop/thoracic_seg/test_transform.nii').get_fdata())

print(new_prediction.shape)

plt.figure(figsize=(16, 8))
plt.subplot(141)
plt.title('Testing Image')
plt.imshow(raw_image_dataset[-1][:,:,raw_image_dataset[-1].shape[2]//2], cmap='gray')
plt.subplot(142)
plt.title('Testing Image')
plt.imshow(segmentation_dataset[-1][:,:,raw_image_dataset[-1].shape[2]//2], cmap='gray')
plt.subplot(143)
plt.title('Testing Label')
plt.imshow(new_prediction[:,:, raw_image_dataset[0].shape[2]//2], cmap='gray')
plt.show()
plt.close()