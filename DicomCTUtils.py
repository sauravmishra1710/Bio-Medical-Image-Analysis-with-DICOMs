import os
import pydicom
import numpy as np
import pandas as pd
from math import ceil

import matplotlib as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology
import seaborn as sns
from sklearn.cluster import KMeans

import scipy
from scipy import ndimage
from skimage import morphology


class DcmCTUtils():
    
    def __init__(self):
        pass
    
    def readdcm(self, filename):
        
        '''
        Read and parse a DICOM dataset stored 
        in the DICOM File Format.
        
        Args:
            filename: path to the dicom file. 
            
        Returns:
            dicomData: return a dataset stored in accordance 
                       with the DICOM File Format
        '''
        
        dicomData = pydicom.dcmread(filename)
        return dicomData
    
    def plot_image(self, image, cm = plt.bone()):
        
        '''
        Plots the dicom image in a figure.
        
        Args:
            image: the dicom format pixel array of the image.
            cm: a colormap instance to be applied on the image.
                Default value plt.bone()
        
        Return:
            None
        
        '''
        plt.figure(dpi=200)
        plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(cm)
        plt.pcolormesh(image)
        
        
    def transform_to_hu(self, dicomData):
    
        """
        transforms a dicom pixel data to the Hounsfield units (HU) 
        scale and returns the transformed HU scale image.
        
        Args:
            dicomData: the dataset in accordance 
                       with the DICOM File Format
                       
        Return:
            hu_image: the image in the hu scale.

        """
        intercept = dicomData.RescaleIntercept
        slope = dicomData.RescaleSlope
        pixel_array = dicomData.pixel_array

        hu_image = pixel_array * slope + intercept

        return hu_image
    
    def window_image(self, image, window_center, window_width):
    
        """
        returns patch of the image as specified by the
        window using the window_center and window_width.
        
        Args:
            image: the dicom format pixel array of the image.
            window_center: the midpoint of the range of the CT numbers displayed.
            window_width: measure of the range of CT numbers that the image contains.
        
        Return:
            window_image: patch of the image as specified by the window

        """

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max

        return window_image
    
    def load_slices(self, path):
    
        """
        loads the different slices of the 
        Chest CT scan.
        
        Args:
            path: the path of the dicom file
            
        Return:
            slices: array of slices in the CT scan.
            
        """
        filenames = os.listdir(path)
        slices = [pydicom.dcmread(f'{path}/{file}') for file in filenames]
        slices.sort(key = lambda x: int(x.InstanceNumber), reverse=True)
        return slices
    
    def get_all_pixel_data(self, dicomData):
    
        '''
        returns the pixel data for all slices
        in the particular CT scan.
        
        Args:
            dicomData: the dataset in accordance 
                       with the DICOM File Format
                       
        Return:
            all_pixel_data: pixel data for all slices.
            
        '''

        all_pixel_data = []

        for ct_slice in dicomData:
            all_pixel_data.append(ct_slice.pixel_array)

        return all_pixel_data
    
    def DisplayCTSlices(self, all_pixel_data, nSlices):
        
        '''
        Displays all the slices 
        in the CT Scan.
        
        Args:
            all_pixel_data: pixel data for all slices.
            nSlices: total number of slices in the CT.
                       
        Return:
            None

        '''

        cols = 6
        rows = int(ceil(nSlices / cols))

        total_subplots = rows * cols
        unused_plots = total_subplots - nSlices

        plt.tight_layout()
        fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = [20, 20])
        fig.patch.set_facecolor('white')

        for idx, image in enumerate(all_pixel_data):
            ax[int(idx/cols), int(idx % cols)].imshow(image, cmap='bone')
            ax[int(idx/cols),int(idx % cols)].set_title("Slice " + str(idx + 1))
            ax[int(idx/cols),int(idx % cols)].axis('off')

        for idx in range(1, unused_plots + 1):
            ax.flat[-idx].set_visible(False)
    
    def transform_all_to_hu(self, slices):
    
        """
        transforms a dicom pixel data to the HU scale
        and returns the transformed HU scale image.
        
        Args:
            slices: the image slices of all the slice in the CT scan.
            
        Returns:
            all_in_hu: array of the image slices converted to the HU scale.

        """

        image = np.stack([s.pixel_array for s in slices])

        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)
            
            all_in_hu = np.array(image, dtype=np.int16)

        return all_in_hu
    
    def plotCTHistogram(self, raw_pixel_arrays, hu_pixel_arrays):
    
        """
        plots the histogram of the pixel values of the 
        raw and HU scale pixel values.

        Args:
            raw_pixel_arrays: the pixel values of the raw images
            hu_pixel_arrays: the pixel values on the HU scale.

        Returns:
            None

        """
        fig,ax = plt.subplots(1,2,figsize=(20,5))
        fig.patch.set_facecolor('white')

        for i in range(len(raw_pixel_arrays)):
            sns.distplot(raw_pixel_arrays[i].flatten(), ax=ax[0], color='green')
            sns.distplot(hu_pixel_arrays[i].flatten(), ax=ax[1], color='blue')

        ax[0].set_title("Raw pixel array distribution")
        ax[1].set_title("Pixel Array distribution on the HU scale")
        
    def getDicomAttributesDictionary(self, dicomData):
    
        """
        Extracts the dicom data attributes and stores
        them as a key value pair in the dictionary format.
        
        Note: We are not storing the pixel array data
              as part of this dictionary since the pixel array
              will be of large dimension and can take up more space.

        Args:
            dicomData: the dataset in accordance 
                           with the DICOM File Format

        Return:
            dicom attributes in the dictionary format.

        """
        return ({attr : getattr(dicomData, attr) for attr in 
                 dir(dicomData) if attr[0].isupper() and attr not in ['PixelData']})
    
    def getDicomDataframe(self, dicomData):
    
        """
        converts the dicom attributes data into a dataframe and 
        retruns the dataframe.

        Args:
            dicomData: the dataset in accordance 
                           with the DICOM File Format

        Return:
            metadata_df: a dataframe object for the dicom attributes.

        """
        dicom_metadata = []

        for dicom_slice in dicomData:
            record = self.getDicomAttributesDictionary(dicom_slice)
            raw_img = dicom_slice.pixel_array
            pixelspacing_r, pixelspacing_c = dicom_slice.PixelSpacing[0], dicom_slice.PixelSpacing[1]
            row_distance = pixelspacing_r * dicom_slice.Rows
            col_distance = pixelspacing_c * dicom_slice.Columns
            record.update({'raw_min':raw_img.min(),
                        'raw_max':raw_img.max(),
                        'raw_mean':raw_img.mean(),
                        'raw_std':raw_img.std(),
                        'raw_diff':raw_img.max() - raw_img.min(),
                        'pixel_spacing_area':pixelspacing_r * pixelspacing_c,
                        'img_area':dicom_slice.Rows * dicom_slice.Columns,
                        'pixel_row_distance':row_distance,
                        'pixel_col_distance':col_distance,
                        'slice_area_cm2':(0.1 * row_distance) * (0.1 * col_distance),
                        'slice_vol_cm3':(0.1 * dicom_slice.SliceThickness) * (0.1 * row_distance) * (0.1 * col_distance)})

            dicom_metadata.append(record)

        metadata_df = pd.DataFrame(dicom_metadata)
        instance_col = metadata_df.pop('InstanceNumber')
        metadata_df.insert(0, 'InstanceNumber', instance_col)
        metadata_df.sort_values("InstanceNumber", inplace = True)
        metadata_df.set_index('InstanceNumber', inplace = True)
        metadata_df.to_pickle('metadata_df.pkl')

        return metadata_df
    
    def resampleDicomPixelArray(self, image, DicomScans, new_spacing=[1,1,1]):
    
        """
        Resamples all the pixel arrays (slices) to a certain isotropic resolution
        which removes the zoom/slice thickness invariance.
        
        Usecase Example @ https://stackoverflow.com/a/42686668/599456
        
        Args:
            image: the images array in the HU scale.
            DicomScans: the dicom scans array for all slices.

        Return:
            metadata_df: a dataframe object for the dicom attributes.
        """
        
        # Determine current pixel spacing
        # spacing = np.array([DicomScans[0].SliceThickness] + list(DicomScans[0].PixelSpacing), dtype=np.float32)
        spacing = np.array([float(DicomScans[0].SliceThickness)] + [float(x) for x in DicomScans[0].PixelSpacing], dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        # specify the array of zoom factors for each axis
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

        return image, new_spacing
    
    def plotScanIn3D(self, image, threshold = 600):
    
        """
        plots a 3D image of the scan
        by parsing through all the resampled 
        slices of the CT Scan.

        This uses the marching_cubes_classic algorithm 
        to plot the lung image in 3D.
        
        This is the general approach to any 3D plotting.
    
        Args:
            image: the resampled set of scans/CT slices.
            threshold: the HU scale value for lungs. Default is set to 600.
            
        Return:
            None
            
        Reference: 
            [1] https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html
            [2] https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial/notebook#3D-plotting-the-scan

        """

        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)

        # Classic marching cubes algorithm to find surfaces in 3d volumetric data.
        # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_classic
        # https://www.kite.com/python/docs/skimage.measure.marching_cubes_classic
        verts, faces = measure.marching_cubes_classic(p, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        plt.show()
        
    def createLungSegment(self, image, display=False):
    
        """
        Segments the lungs from other structures.

        Args:
            image: the image to be segmented.
            display: a boolean flag set to display the various lung segments.

        Return:
            overlayed_image: the final overlayed image where the mask is applied on the original image.


        Reference: 
            https://github.com/booz-allen-hamilton/DSB3Tutorial/blob/master/tutorial_code/LUNA_segment_lung_ROI.py

        """
        row_size= image.shape[0]
        col_size = image.shape[1]
        mean = np.mean(image)
        std = np.std(image)
        image = image - mean
        image = image / std

        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = image[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
        mean = np.mean(middle)  
        img_max = np.max(image)
        img_min = np.min(image)

        # To improve threshold finding, I'm moving the 
        # underflow and overflow on the pixel spectrum
        image[image == img_max] = mean
        image[image == img_min] = mean

        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)

        # Threshold the image and the output will be a binary image. Morphology workes either on binary or gray images.
        thresh_img = np.where(image < threshold, 1.0, 0.0)

        # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
        # We don't want to accidentally clip the lung.
        eroded = morphology.erosion(thresh_img, np.ones([3,3]))
        dilation = morphology.dilation(eroded, np.ones([8,8]))

        # Label each region and obtain the region properties. The background region is removed by removing regions 
        # with a bbox that is to large in either dimnsion. Also, the lungs are generally far away from the top 
        # and bottom of the image, so any regions that are too close to the top and bottom are removed
        # This does not produce a perfect segmentation of the lungs from the image, but it is surprisingly 
        # good considering its simplicity. 
        labels = measure.label(dilation) # Different labels are displayed in different colors
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)

        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2] - B[0] < row_size / 10*9 and B[3]-B[1] < col_size/10*9 and B[0] > row_size/5 and B[2] < col_size/5*4:
                good_labels.append(prop.label)

        mask = np.ndarray([row_size,col_size], dtype=np.int8)
        mask[:] = 0

        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

        overlayed_image = mask * image

        if display:

            fig, ax = plt.subplots(3, 2, figsize=[12, 12])
            fig.patch.set_facecolor('white')

            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(image, cmap='gray')
            ax[0, 0].axis('off')

            ax[0, 1].set_title("Threshold")
            ax[0, 1].imshow(thresh_img, cmap='gray')
            ax[0, 1].axis('off')

            ax[1, 0].set_title("After Erosion and Dilation")
            ax[1, 0].imshow(dilation, cmap='gray')
            ax[1, 0].axis('off')

            ax[1, 1].set_title("Color Labels")
            ax[1, 1].imshow(labels)
            ax[1, 1].axis('off')

            ax[2, 0].set_title("Final Mask")
            ax[2, 0].imshow(mask, cmap='gray')
            ax[2, 0].axis('off')

            ax[2, 1].set_title("Apply Mask on Original")
            ax[2, 1].imshow(overlayed_image, cmap='gray')
            ax[2, 1].axis('off')

            plt.show()

        return overlayed_image