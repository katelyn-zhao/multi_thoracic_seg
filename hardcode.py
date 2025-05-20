all_data = pd.read_excel('all_stats.xlsx')
all_data = all_data[3:]
all_data.columns = (['image_name',
'ground_truth_1_mean', 'ground_truth_1_std', 
'ground_truth_2_mean', 'ground_truth_2_std', 
'ground_truth_3_mean', 'ground_truth_3_std', 
'ground_truth_4_mean', 'ground_truth_4_std', 
'ground_truth_5_mean', 'ground_truth_5_std', 
'unet_1_mean', 'unet_1_std',
'unet_2_mean', 'unet_2_std',
'unet_3_mean', 'unet_3_std',
'unet_4_mean', 'unet_4_std',
'unet_5_mean', 'unet_5_std',
'unet++_1_mean', 'unet++_1_std',
'unet++_2_mean', 'unet++_2_std',
'unet++_3_mean', 'unet++_3_std',
'unet++_4_mean', 'unet++_4_std',
'unet++_5_mean', 'unet++_5_std'])
all_data['unet_difference_1'] = np.abs(all_data['unet_1_mean'] - all_data['ground_truth_1_mean'])
all_data['unet_difference_2'] = np.abs(all_data['unet_2_mean'] - all_data['ground_truth_2_mean'])
all_data['unet_difference_3'] = np.abs(all_data['unet_3_mean'] - all_data['ground_truth_3_mean'])
all_data['unet_difference_4'] = np.abs(all_data['unet_4_mean'] - all_data['ground_truth_4_mean'])
all_data['unet_difference_5'] = np.abs(all_data['unet_5_mean'] - all_data['ground_truth_5_mean'])
all_data['unet_std_difference_1'] = np.abs(all_data['unet_1_std'] - all_data['ground_truth_1_std'])
all_data['unet_std_difference_2'] = np.abs(all_data['unet_2_std'] - all_data['ground_truth_2_std'])
all_data['unet_std_difference_3'] = np.abs(all_data['unet_3_std'] - all_data['ground_truth_3_std'])
all_data['unet_std_difference_4'] = np.abs(all_data['unet_4_std'] - all_data['ground_truth_4_std'])
all_data['unet_std_difference_5'] = np.abs(all_data['unet_5_std'] - all_data['ground_truth_5_std'])

all_data['unet++_difference_1'] = np.abs(all_data['unet++_1_mean'] - all_data['ground_truth_1_mean'])
all_data['unet++_difference_2'] = np.abs(all_data['unet++_2_mean'] - all_data['ground_truth_2_mean'])
all_data['unet++_difference_3'] = np.abs(all_data['unet++_3_mean'] - all_data['ground_truth_3_mean'])
all_data['unet++_difference_4'] = np.abs(all_data['unet++_4_mean'] - all_data['ground_truth_4_mean'])
all_data['unet++_difference_5'] = np.abs(all_data['unet++_5_mean'] - all_data['ground_truth_5_mean'])
all_data['unet++_std_difference_1'] = np.abs(all_data['unet++_1_std'] - all_data['ground_truth_1_std'])
all_data['unet++_std_difference_2'] = np.abs(all_data['unet++_2_std'] - all_data['ground_truth_2_std'])
all_data['unet++_std_difference_3'] = np.abs(all_data['unet++_3_std'] - all_data['ground_truth_3_std'])
all_data['unet++_std_difference_4'] = np.abs(all_data['unet++_4_std'] - all_data['ground_truth_4_std'])
all_data['unet++_std_difference_5'] = np.abs(all_data['unet++_5_std'] - all_data['ground_truth_5_std'])

differences = all_data[['image_name', 'unet_difference_1', 'unet_difference_2', 'unet_difference_3', 'unet_difference_4', 'unet_difference_5', 'unet++_difference_1', 'unet++_difference_2', 'unet++_difference_3', 'unet++_difference_4', 'unet++_difference_5', 'unet_std_difference_1', 'unet_std_difference_2', 'unet_std_difference_3', 'unet_std_difference_4', 'unet_std_difference_5', 'unet++_std_difference_1', 'unet++_std_difference_2', 'unet++_std_difference_3', 'unet++_std_difference_4', 'unet++_std_difference_5']]
differences.set_index('image_name', inplace=True)
differences.mean()
