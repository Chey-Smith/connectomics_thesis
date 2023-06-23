import os
import numpy as np
import nibabel as nib
from nibabel.streamlines import load
from nibabel.affines import apply_affine
from ants import registration, apply_transforms, ANTsImage, image_write, image_read
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
from numpy.random import randn

MNI_fname = "/Data/users/lars/MNI152_T1_1mm_brain.nii.gz"
MNI_ants = image_read(MNI_fname)

atlas_names = ["AAL", "Brainnetome", "Schaefer", "Gordon", "DK", "Yeo"]
atlas_files = ["/home/lars/Data__lars/atlases/AAL1/atlas/AAL_90.nii",
               "/home/lars/Data__lars/atlases/Brainnetome/BN_Atlas_246_2mm.nii",
               "no",
               "/home/lars/Data__lars/atlases/Gordon2016/Parcels/Parcels_MNI_111.nii",
               "/home/lars/Data__lars/atlases/Desikan-Killiany/DK_atlas.nii",
               "/home/lars/Data__lars/atlases/Yeo/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz"]

atlas_sizes = [90, 246, 400, 333, 72, 114]

atlas_index = 5

atlas_name = atlas_names[atlas_index]
atlas_file = atlas_files[atlas_index]

atlas_ants = image_read(atlas_file)

print(atlas_name)
print(atlas_file)

atlas = nib.load(atlas_file)
atlas_data = atlas.get_fdata().astype(int)

expected_atlas_size = atlas_sizes[atlas_index]
atlas_size = np.max(atlas_data)

if atlas_size != expected_atlas_size:
    raise Exception("Number of atlas regions is incorrect, expected %s but found %s regions for %s atlas" %
                    (expected_atlas_size, atlas_size, atlas_name))

overwrite = False

data_path = "/home/lars/Data__lars/HCP_1200/"
print(data_path)
i = 0

subj_list = os.listdir(data_path)

for subj in subj_list:
    i += 1
    print(subj)
    print(str(i) + "/" + str(len(subj_list)))
    subj_path = os.path.join(data_path, subj)

    # preprocess the raw DWI scan
    if not os.path.exists(os.path.join(subj_path, "preprocessed_DWI.nii.gz")):
        bvec_file = os.path.join(subj_path, "bvecs.bvec")
        bval_file = os.path.join(subj_path, "bvals.bval")
        os.system("dwifslpreproc %s %s -fslgrad %s %s -rpe_none -pe_dir AP" %
                                        (os.path.join(subj_path, "DWI.nii.gz"),
                                        os.path.join(subj_path, "preprocessed_DWI.nii.gz"),
                                         bvec_file, bval_file))

    # Generate the FOD from the preprocessed DWI
    if not os.path.exists(os.path.join(subj_path, "FOD.mif")):
        dwi_file = os.path.join(subj_path, "preprocessed_DWI.nii.gz")
        mask_file = os.path.join(subj_path, "mask.nii.gz")
        mask_dilated_file = os.path.join(subj_path, "mask_dilated.nii.gz")
        bvec_file = os.path.join(subj_path, "bvecs.bvec")
        bval_file = os.path.join(subj_path, "bvals.bval")
        wm_response_file = os.path.join(subj_path, "wm_response.txt")
        fod_file = os.path.join(subj_path, "FOD.mif")

        if not os.path.exists(bvec_file):
            bvec_file = os.path.join(subj_path, "bvecs.nii")

        if not os.path.exists(bval_file):
            bval_file = os.path.join(subj_path, "bvals.nii")

        print("Extracting mask from DWI...")
        os.system("dwi2mask %s %s -fslgrad %s %s -force" % (dwi_file, mask_file, bvec_file, bval_file))

        print("dilating mask...")
        os.system("fslmaths %s -kernel gauss 1.5 -dilF %s" % (mask_file, mask_dilated_file))

        print("Estimating response function...")
        os.system("dwi2response tournier %s %s -mask %s -fslgrad %s %s -force" %
                          (dwi_file, wm_response_file, mask_dilated_file, bvec_file, bval_file))

        print("Estimating FOD...")
        os.system("dwi2fod csd %s %s %s -mask %s -fslgrad %s %s -force" %
                          (dwi_file, wm_response_file, fod_file, mask_dilated_file, bvec_file, bval_file))

    # register atlas to DWI space
    registered_atlas_file = os.path.join(subj_path, "%s_dwireg.nii" % atlas_name)
    if not os.path.exists(registered_atlas_file) or overwrite:
        #extract b0
        preprocessed_dwi_file = os.path.join(subj_path, "preprocessed_DWI.nii.gz")
        bvec_file = os.path.join(subj_path, "bvecs.bvec")
        bval_file = os.path.join(subj_path, "bvals.bval")

        if not os.path.exists(bvec_file):
            bvec_file = os.path.join(subj_path, "bvecs.nii")
        if not os.path.exists(bval_file):
            bval_file = os.path.join(subj_path, "bvals.nii")

        b0_file = os.path.join(subj_path, "b0.nii")
        if not os.path.exists(b0_file) or overwrite:
            print("extracting b0...")
            os.system("dwiextract %s -fslgrad %s %s extracted_DWI.nii -bzero -force" % (preprocessed_dwi_file, bvec_file, bval_file))
            os.system("mrmath extracted_DWI.nii mean %s -axis 3 -force" % b0_file)

	# skull-strip b0
        stripped_b0_file = os.path.join(subj_path, "b0_stripped.nii.gz")
        if not os.path.exists(stripped_b0_file) or overwrite:
            print("skull-stripping b0...")
            os.system("bet2 %s %s -f 0.2" % (b0_file, stripped_b0_file))

	# register atlas using ANTS
        print("registering atlas...")
        stripped_b0 = image_read(stripped_b0_file)
        reg_result = registration(stripped_b0, MNI_ants)
        mni_dwireg = reg_result["warpedmovout"]
        transform = reg_result["fwdtransforms"]
        atlas_dwireg_ants = apply_transforms(stripped_b0, atlas_ants, transform, interpolator='genericLabel')
        os.remove(registered_atlas_file)
        image_write(atlas_dwireg_ants, registered_atlas_file)

        # clean up temp files
        for f in os.listdir("/tmp"):
            if f.startswith("tmp") and "GenericAffine.mat" in f:
                os.remove(os.path.join("/tmp", f))
            if f.startswith("tmp") and "Warp.nii.gz" in f:
                os.remove(os.path.join("/tmp", f))

    # generate structural matrix
    struct_mat_file = os.path.join(subj_path, "structural_matrix_%s.npy" % atlas_name)
    if not os.path.exists(struct_mat_file) or overwrite:

        print("loading DWI...")
        dwi = nib.load(os.path.join(subj_path, "preprocessed_DWI.nii.gz"))
        print("building structural matrix...")
        struct_mat = np.zeros((atlas_size, atlas_size))

        atlas_registered = nib.load(registered_atlas_file)

	# resample atlas data if the shapes don't match
        if dwi.shape[:3] != atlas_registered.shape:
            atlas_registered = resample_to_img(atlas_registered, dwi, interpolation="nearest")

        atlas_data_registered = atlas_registered.get_fdata()
        endpoints_file = os.path.join(subj_path, "streamlines_sarwar_endpoints.npy")
        
	# process endpoints into connectivity matrix
        if os.path.exists(endpoints_file):
            endpoints = np.load(endpoints_file)
            for endpoint in endpoints:
                start = nib.affines.apply_affine(np.linalg.inv(dwi.affine), endpoint[0]).astype(int)
                end = nib.affines.apply_affine(np.linalg.inv(dwi.affine), endpoint[1]).astype(int)

                start_val = round(atlas_data_registered[start[0], start[1], start[2]])
                end_val = round(atlas_data_registered[end[0], end[1], end[2]])

                if start_val != 0 and end_val != 0 and start_val != end_val:
                    struct_mat[start_val - 1, end_val - 1] += 1
                    struct_mat[end_val - 1, start_val - 1] += 1

        np.save(os.path.join(subj_path, "structural_matrix_%s.npy" % atlas_name), struct_mat)

	# apply Gaussian resampling to the matrix
	num_values = int((atlas_size - 1) * (atlas_size) / 2)
    	gaussians = randn(num_values)
   	gaussians.sort()

    	struct_mat = np.load(os.path.join(subj_path, "structural_matrix_%s.npy" % atlas_name))
    	struct_mat_u = struct_mat[np.triu_indices(atlas_size, k=1)]

    	sort_indices = np.argsort(struct_mat_u)

    	new_u = np.zeros(num_values)
    	for j in range(num_values):
        	new_u[sort_indices[j]] = gaussians[j]
        
    	resampled_mat = np.zeros((atlas_size, atlas_size))
    	j = 0
    	for x in range(atlas_size):
        	for y in range(x + 1, atlas_size):
            		resampled_mat[x, y] = new_u[j]
            		resampled_mat[y, x] = new_u[j]
            		j += 1

    	resampled_mat /= np.std(resampled_mat) * 10
    	resampled_mat = resampled_mat - np.mean(resampled_mat) + 0.5

    	np.save(os.path.join(subj_path, "structural_matrix_%s_resampled.npy" % atlas_name), resampled_mat)
