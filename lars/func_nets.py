import nibabel as nib
import numpy as np
from nilearn.image.resampling import resample_to_img
import matplotlib.pyplot as plt
from nilearn.plotting import plot_matrix
import os
from scipy.stats import pearsonr

def fast_corr_mat(time_series):
    t_demeaned = time_series - np.average(time_series, axis=1).reshape((-1, 1))

    t_z = t_demeaned / np.std(t_demeaned, axis=1).reshape((-1, 1))

    N = len(time_series)
    time_length = len(time_series[0])

    corr_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i):
            corr_mat[i, j] = np.sum(t_z[i] * t_z[j]) / (time_length - 1)
            corr_mat[j, i] = corr_mat[i, j]

    return corr_mat

atlas_names = ["AAL", "Brainnetome", "Schaefer", "Gordon", "DK", "Yeo"]
atlas_files = ["/home/lars/Data__lars/atlases/AAL1/atlas/AAL_90.nii",
               "/home/lars/Data__lars/atlases/Brainnetome/BN_Atlas_246_2mm.nii",
               "no",
               "/home/lars/Data__lars/atlases/Gordon2016/Parcels/Parcels_MNI_111.nii",
               "/home/lars/Data__lars/atlases/Desikan-Killiany/DK_atlas.nii",
               "/home/lars/Data__lars/atlases/Yeo/1000subjects_reference/Yeo_JNeurophysiol11_SplitLabels/MNI152/Yeo2011_17Networks_N1000.split_components.FSL_MNI152_1mm.nii.gz"]

atlas_sizes = [90, 246, 400, 333, 68, 114]

atlas_index = 5

atlas_name = atlas_names[atlas_index]
atlas_file = atlas_files[atlas_index]

print(atlas_name)
print(atlas_file)

atlas = nib.load(atlas_file)

data_path = "/home/lars/Data__lars/HCP_1200/"
overwrite = False
i = 0

# resample atlas to HCP
atlas = resample_to_img(atlas, nib.load(os.path.join(os.path.join(data_path, "130619"), "preprocessed_rfMRI_sess_1_2_concat.nii.gz")),
                        interpolation="nearest")

atlas_data = atlas.get_fdata().astype(int)
expected_atlas_size = atlas_sizes[atlas_index]
atlas_size = np.max(atlas_data)


if atlas_size != expected_atlas_size:
    raise Exception("Number of atlas regions is incorrect, expected %s but found %s regions for %s atlas" %
                    (expected_atlas_size, atlas_size, atlas_name))

for subj in os.listdir(data_path):
    subj_path = os.path.join(data_path, subj)
    i += 1
    print(subj)
    print(str(i) + "/1052")

    concat_fmri = os.path.join(subj_path, "preprocessed_rfMRI_sess_1_2_concat.nii.gz")
    if not os.path.exists(concat_fmri):
        sess_1 = nib.load(os.path.join(subj_path, "preprocessed_rfMRI_sess_1.nii.gz"))
        sess_2 = nib.load(os.path.join(subj_path, "preprocessed_rfMRI_sess_2.nii.gz"))

        sess_1_data = sess_1.get_fdata()
        time_avgs = np.tile(np.mean(sess_1_data, axis=3).reshape((91, 109, 91, 1)), (1, 1, 1, 1200))
        sess_1_demeaned = sess_1_data - time_avgs

        sess_2_data = sess_2.get_fdata()
        time_avgs = np.tile(np.mean(sess_2_data, axis=3).reshape((91, 109, 91, 1)), (1, 1, 1, 1200))
        sess_2_demeaned = sess_2_data - time_avgs

        concat = np.concatenate((sess_1_demeaned, sess_2_demeaned), axis=3)
        concat_nifti = nib.Nifti1Image(concat, sess_1.affine, sess_1.header)
        nib.save(concat_nifti, concat_fmri)
        os.system("gzip %s" % concat_fmri)

    if not os.path.exists(os.path.join(subj_path, "time_series_%s.npy" % (atlas_name))) or overwrite:
        print("loading resting-state...")
        resting_state_file = os.path.join(subj_path, "preprocessed_rfMRI_sess_1_2_concat.nii.gz")
        resting_state = nib.load(resting_state_file)

        resting_state_data = resting_state.get_fdata()

        time_length = resting_state_data.shape[3]

        time_series = np.zeros((atlas_size, time_length))

        print("calculating ROI time series...")
        for r in range(1, atlas_size + 1):
            region_voxels = np.argwhere(atlas_data == r)
            time_series[r - 1] = np.average(resting_state_data[region_voxels[:, 0], region_voxels[:, 1], region_voxels[:, 2]],
                                            axis=0)

        np.save(os.path.join(subj_path, "time_series_%s.npy" % (atlas_name)), time_series)

    if not os.path.exists(os.path.join(subj_path, "functional_matrix_%s.npy" % (atlas_name))) or overwrite:
        time_series = np.load(os.path.join(subj_path, "time_series_%s.npy" % (atlas_name)))
        corr_mat = fast_corr_mat(time_series)

        np.save(os.path.join(subj_path, "functional_matrix_%s.npy" % (atlas_name)), corr_mat)