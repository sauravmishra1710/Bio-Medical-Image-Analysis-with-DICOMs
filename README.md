# Bio-Medical-Image-Analysis-with-DICOMs

## What is medical imaging?

Medical imaging refers to techniques, standards and procedures that are used for imaging the interior of a body for clinical analysis and medical intervention, as well as visual representation of the function of some organs or tissues. Medical imaging seeks to reveal internal structures hidden by the skin and bones, as well as to diagnose and treat disease. Medical imaging also establishes a database of normal anatomy and physiology to make it possible to identify abnormalities.  

Medical imaging is very much essential and helpful in almost all medical setting and levels of healthcare helping physicians to arrive at more accurate diagnoses, prognoses and take correct treatment decisions.

Medical Imaging includes the following radiological imaging techniques such as:

- X-ray radiography
- Fluoroscopy
- Magnetic resonance imaging (MRI)
- Medical ultrasonography or ultrasound
- Endoscopy
- Elastography
- Tactile imaging
- Thermography


**Reference:** https://en.wikipedia.org/wiki/Medical_imaging

## Biomedical Imaging Coordinate System

Coordinate system is used for identifying the location of a point. Three types coordinate systems commonly used in imaging applications: 
- **World Coordinate**,
- **Anatomical Coordinate** (also called the Patient Coordinate System) and 
- **Medical Image Coordinate** systems.

![medical_coordinate_system.PNG](https://github.com/sauravmishra1710/Bio-Medical-Image-Analysis-with-DICOMs/blob/main/images/medical_coordinate_system.PNG)


**Image Source** @ https://www.slicer.org/wiki/Coordinate_systems

### World Coordinate System

*The world coordinate system is typically a Cartesian coordinate system in which a model (e.g. a MRI scanner or a patient) is positioned. Every model has its own coordinate system but there is only one world coordinate system to define the position and orientation of each model.*

### Anatomical Coordinate System

*The most important model coordinate system for medical imaging techniques is the anatomical space. This space consists of **three planes** to describe the standard anatomical position of a human:*

***Axial Plane:*** *The axial plane is parallel to the ground and a slice near the head is referred to as **superior** while a slice closer to the feet, is known as the **inferior**. The axial plane is actually when you place point of view above the patient and look down. Depending on the region of the 3D medical image you will observe different anatomical structures.* The image below shows the Axial View of a CT scan of the chest - 

![Axial_View_CTChest.PNG](https://github.com/sauravmishra1710/Bio-Medical-Image-Analysis-with-DICOMs/blob/main/images/Axial_View_CTChest.PNG)

***Coronal Plane:*** *The coronal plane is perpendicular to the ground and separates the front from the back. In this, the point of view is either in front of eyes(**anterior plane**) or back of the patient(**posterior plane**).* The image below shows the Coronal View of a CT scan of the chest - 

![Coronal_View_CTChest.PNG](https://github.com/sauravmishra1710/Bio-Medical-Image-Analysis-with-DICOMs/blob/main/images/Coronal_View_CTChest.PNG)

***Sagittal Plane:*** *The sagittal plane separates the Left from the Right. Basically, this is a side view. Instead of looking from above the patient, now we look from the side. The side can be either right or left. Which side and direction is the positive one, depends on the coordinate system.* The image below shows the Sagittal View of a CT scan of the chest -

![Sagittal_View_CTChest.PNG](https://github.com/sauravmishra1710/Bio-Medical-Image-Analysis-with-DICOMs/blob/main/images/Sagittal_View_CTChest.PNG)

*The anatomical coordinate system is a continuous three-dimensional space in which an image has been sampled. In neuroimaging, it is common to define this space with respect to the human whose brain is being scanned. Hence the 3D basis is defined along the anatomical axes of anterior-posterior, inferior-superior, and left-right.*

**Image Source** @ All the above CT Chest images have been taken from [3D Slicer Smaple Data](https://slicer.kitware.com/midas3/item/126145) and viewed in the [3D slicer software](https://www.slicer.org/wiki/Main_Page)

### Medical Image Coordinate System (Voxel Space)

*The image coordinate system describes **how an image was acquired with respect to the anatomy**. Medical scanners create 3D arrays/tensors of points and cells which start at the upper left corner. `The i axis increases to the right, the j axis to the bottom and the k axis backwards.`*

*In addition to the intensity value of each voxel (i, j, k), the origin and spacing of the anatomical coordinates are stored in the form of metadata. `The origin represents the position of the first voxel (0,0,0) in the anatomical coordinate system and the spacing specifies the distance between voxels along each axis`. Using the origin and spacing, the corresponding position of each (image coordinate) voxel in anatomical coordinates can be calculated.*

*`Voxel spacing is basically the real size of the voxel that corresponds to a 3D region and the distance between two voxels.`*

**References:** 

[1] https://www.slicer.org/wiki/Coordinate_systems#World_coordinate_system

[2] https://theaisummer.com/medical-image-coordinates/

[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3354356/ (Managing DICOM images: Tips and tricks for the radiologist)


## DICOM

**DICOM** defines standards for storing data in memory and on disk, and for communicating this data between machines over a network. DICOM files are binary dumps of the objects in memory that DICOM sends across the network.

A DICOM image file is an outcome of the **Digital Imaging and COmmunications in Medicine** standard and represented as **`.dcm`**. It is a standard, internationally accepted format to view, store, retrieve and share medical images. DICOM differs from other image formats because it groups information into datasets. DICOM file consist of **header** and **image data** collectively in one file. 

The header metadata consists of data that describes the image, and the patient data. Patient data could include information about the patient’s demographic information like name, age, gender, and date of birth. The metadata may also contain information about image characteristics such as acquisition parameters, pixel intensity, matrix size, and dimensions of the image, type of the image (CT, MRI, etc..), the context of the imaging study, etc... These metadata information are coded onto the image so that the patient to whom the image belongs can be easily identified.

**DICOM Communications Protocol -**
DICOM not only defines the image and metadata but also the transport layer protocol that enables easy communication and passage of information through the network. The entire standard is made up of multiple related but independent sections. Nevertheless, all medical imaging applications that are connected to the hospital network use the DICOM protocol to exchange information, which is mainly DICOM images.

## Pydicom and NiBabel

Medical image data are coded in 2 formats, **DICOM** and **NIfTI** formats are used. 
- **DICOM** format has multiple 2d image slices which together form a 3d image.
- **NIfTI** format has only one file that contains the 3d image. 

**Pydicom** and **NiBabel** are libraries to dela with the DICOM and NIfTI images respectively. Both libraries load the images in a numpy array format, which makes it easy to extract and save 2d images in the jpg format.

## Analyze DICOM Images

**pydicom** a Python package developed for working with DICOM files and images contains APIs for reading, viewing, modifying the dcm files and objects. Pydicom provides an interface to read the complex DICOM files for easy interpretation.
It contains APIs to manipulate image and metadata contained in the DICOM files.
