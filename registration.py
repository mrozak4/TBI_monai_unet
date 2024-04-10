from numpy import arcsin, cos, arctan2, degrees, float32
from skimage.io import imread, imsave
from ants import from_numpy, registration, apply_transforms, read_transform
from re import sub
from os.path import basename, dirname
from PIL.TiffTags import TAGS
from tifffile import TiffFile

def _ants_affine_to_distance(affine):
    """
    Convert ANTs affine transformation matrix to distance and rotation values.

    Args:
        affine (array-like): The ANTs affine transformation matrix.

    Returns:
        tuple: A tuple containing the distance and rotation values in the order (dx, dy, dz, rot_x, rot_y, rot_z).
    """

    # Extract translation values
    dx, dy, dz = affine[9:]

    # Extract rotation values
    rot_x = arcsin(affine[6])
    cos_rot_x = cos(rot_x)
    rot_y = arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    # Convert rotation values to degrees
    deg = degrees

    return dx, dy, dz, deg(rot_x), deg(rot_y), deg(rot_z)

def register_paired_images(fix_file, mov_files, out_dir, sigma=2, flip = False):
    """
    Register paired images using ANTs registration.

    Args:
        fix_file (str): Path to the fixed image file.
        mov_files (list): List of paths to the moving image files.
        out_dir (str): Output directory to save the registered images.
        sigma (float, optional): Sigma value for the registration. Defaults to 2.
        flip (bool, optional): Whether to flip the moving image. Defaults to False.

    Returns:
        list: List of displacement values for each registered image.

    Raises:
        FileNotFoundError: If any of the input image files are not found.
    """
    
    print(flip)
    
    # Read the fixed image
    fix_numpy = imread(fix_file)
    fix = from_numpy(float32(fix_numpy[:,0])) # Convert images to ants 

    if flip == True:
        # Extract metadata from the fixed image
        with TiffFile(fix_file) as tif:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            image = tif.pages[0].asarray()
        #print(tif_tags)
        start_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis startPosition #1' in x][0].split(' ')[-1])
        end_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis endPosition #1' in x][0].split(' ')[-1])
        direction = end_str - start_str
        if [x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'imagingParam zDriveUnitType' in x][0].split(' ')[-1] == 'Piezo':
            direction = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'acquisitionValue zPosition' in x][0].split(' ')[-1])
            piezo = True
        else:
            piezo = False
        #print(direction)
        
    res2 = []
    for mov_file in mov_files:
        # Read the moving image
        mov_numpy = imread(mov_file)
        
        if flip == True and piezo == False:
            with TiffFile(mov_file) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                image = tif.pages[0].asarray()
            start_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis startPosition #1' in x][0].split(' ')[-1])
            end_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis endPosition #1' in x][0].split(' ')[-1])
            #print((end_str - start_str))
            # Flip the moving image if necessary
            if direction * (end_str - start_str) < 0:
                mov_numpy = mov_numpy[::-1]
                print('flipped')
        elif flip == True and piezo == True:
            with TiffFile(mov_file) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                image = tif.pages[0].asarray()
            piezo_dir = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'acquisitionValue zPosition' in x][0].split(' ')[-1])
            #print((piezo_dir))
            # Flip the moving image if necessary
            if direction != piezo_dir:
                mov_numpy = mov_numpy[::-1]
                print('flipped')

        mov = from_numpy(float32(mov_numpy[:,0]))

        # Register images and get displacement
        mytx = registration(fixed=fix,
                            moving=mov,
                            type_of_transform='Rigid',
                            total_sigma=sigma,
                            aff_metric='meansquares')

        # Move vascular channel
        warpedraw_1 = apply_transforms(fixed=fix,
                                       moving=from_numpy(float32(mov_numpy[:,0])),
                                       transformlist=mytx['fwdtransforms'],
                                       interpolator='linear')

        # Move neuron channel
        warpedraw_2 = apply_transforms(fixed=fix,
                                       moving=from_numpy(float32(mov_numpy[:,1])),
                                       transformlist=mytx['fwdtransforms'],
                                       interpolator='linear')

        # Combine moved channels into one image
        mov_numpy[:,0,:,:] = warpedraw_1[:,:,:]
        mov_numpy[:,1,:,:] = warpedraw_2[:,:,:]

        # Append displacement values to the result list
        res2.append(_ants_affine_to_distance(read_transform(mytx['fwdtransforms'][0]).parameters))

        # Save warped followup image and baseline image
        imsave(out_dir + sub('.tif','_warped.tif',basename(dirname(mov_file)) + '-' + basename(mov_file)),mov_numpy)

    # Save the fixed image
    imsave(out_dir + sub('.tif','_warped.tif',basename(dirname(fix_file)) + '-' + basename(fix_file)),fix_numpy)
    
    return res2