# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df, custom_downsample_per_slide
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import openslide

allpath=[]
allname=[]

def getallfile(path):

	allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
	for file in allfilelist:
		filepath=os.path.join(path,file)
		# 如果是文件夹，递归调用函数
		if os.path.isdir(filepath):
			getallfile(filepath)
		# 如果不是文件夹，保存文件路径及文件名
		elif os.path.isfile(filepath):
			allpath.append(filepath)
			allname.append(file)
	return allpath, allname


def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source_path, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None, region_list = None, data_flag = None, expect_mag = -1):
	

	#TODO: add a list for loading data from multiple folders 
	if region_list is not None:
		slides = []
		full_path = []
		for i in region_list:
			if data_flag == 'cptac':				
				cur_source = os.path.join(source_path, i, i)
				# slides.extend(sorted(os.listdir(cur_source)))
				cur_slides = sorted(os.listdir(cur_source))	
				cur_slides = [slide for slide in cur_slides if os.path.isfile(os.path.join(cur_source, slide))]
				cur_path = [os.path.join(cur_source, slide) for slide in cur_slides if os.path.isfile(os.path.join(cur_source, slide))]
				slides.extend(cur_slides)
				full_path.extend(cur_path)

			elif data_flag == 'tcga':
				cur_source = os.path.join(source_path, i)
				files, names = getallfile(cur_source)
				cur_path = sorted(files)
				cur_path = [abs_path for abs_path in cur_path if abs_path[-4:] == '.svs' ]
				
				# should also get rid of abs pathes of slides that is not tumor
				cur_path = [pathes for pathes in cur_path if int(pathes.split('/',-1)[-1][13]) != 1]

				cur_slides = [slide.split('/',-1)[-1] for slide in cur_path]
				# pick up normal slide. e.g., barcode: TCGA-A7-A6VV-01Z, check the first number of '01Z' whehter 0[tumor] or not [normal] 
				cur_slides = [slide for slide in cur_slides if int(slide[13]) != 1]
				
				slides.extend(cur_slides)
				full_path.extend(cur_path)

	else:
		slides = sorted(os.listdir(source_path))
		slides = [slide for slide in slides if os.path.isfile(os.path.join(source_path, slide))]
		full_path = [os.path.join(source_path, slide) for slide in slides if os.path.isfile(os.path.join(source_path, slide))]
	# TODO: add a script for tcga dataset 
		# full path
		# 

	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, full_path)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, full_path)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i+1, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)


		
		# Inialize WSI
		# full_path = os.path.join(source_path, slide)
		full_path = process_stack.loc[idx, 'full_path']
		WSI_object = WholeSlideImage(full_path)
		
		# modify patch_size and step_size according to the magnification
		slide_mag = WSI_object.objective_power

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			df.loc[idx, 'mag'] = slide_mag
			continue

		if WSI_object.objective_power == 0:
			print('{} do not have magnification information, ignored'.format(slide_id))
			df.loc[idx,'status'] = 'ignore'
			continue

		if expect_mag == -1:
			expect_mag = 40
		
		ratio = custom_downsample_per_slide(slide_mag, expect_mag, default_mag=40, if_patch=True)
		cur_patch_size = int(patch_size / ratio)
		cur_step_size = int(step_size / ratio)
			# if not, it means we only want to change the patch size for each slide with default

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': cur_patch_size, 'step_size': cur_step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

		# TODO: record the original magnification
		df.loc[idx, 'mag'] = slide_mag
		# df.loc[idx, 'patch_size'] =cur_patch_size

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

"""
	flag priority
		-data
		-region list
		-specifi region

"""	
parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')
parser.add_argument('--source_path', type = str,
					help='path to folder containing raw wsi image files'
					,default='/data2/WSI_CPTAC/'
					)

parser.add_argument('--mag', default=-1, type=int, choices=[-1, 40, 20, 10, 5], 
					help='choose the magnification of slides')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')

parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data'
					,default='./data/cptac'
					)

parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')

# if __name__ == '__main__':
    
current_path = os.path.abspath(__file__)

args = parser.parse_args()

args.preset = 'tcga.csv'

if args.data == 'cptac':
	args.source_path = '/data2/WSI_CPTAC/'
	region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']

	args.region = region_list if not args.specific_region else [region_list[args.r]]	 
	args.save_dir = './data/cptac' if not args.specific_region else './data/cptac/' + region_list[args.r]

elif args.data == 'tcga':
	args.source_path = '/data2/WSI_TCGA/'
	region_list = ['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',
        'LGG', 'PRAD', 'THCA', 'SKCM', 'STAD', 'BLCA', 
        'LIHC', 'CESC', 'KIRP', 'SARC', 'ESCA', 'PCPG',
        'READ', 'LAML', 'TGCT', 'THYM', 'ACC', 'MESO', 
        'UVM', 'KICH', 'UCS', 'DLBC', 'CHOL']

	args.region = region_list if not args.specific_region else [region_list[args.r]]	 

	args.save_dir = './data/tcga' if not args.specific_region else './data/tcga/' + region_list[args.r]

else:
	args.region == None

if args.mag != -1:
	# ratio = 40/ args.mag since the max/default magnification is 40 
	args.save_dir = args.save_dir + '/mag{}x_patch{}_fp'.format(args.mag, int(args.patch_size/(40/args.mag)))
else:
	args.save_dir = args.save_dir + '/mag{}x_patch{}_fp'.format(40, int(args.patch_size))

""" add region_list
"""
patch_save_dir = os.path.join(args.save_dir, 'patches')
mask_save_dir = os.path.join(args.save_dir, 'masks')
stitch_save_dir = os.path.join(args.save_dir, 'stitches')

if args.process_list:
	process_list = os.path.join(args.save_dir, args.process_list)

else:
	process_list = None

print('source_path: ', args.source_path)
print('patch_save_dir: ', patch_save_dir)
print('mask_save_dir: ', mask_save_dir)
print('stitch_save_dir: ', stitch_save_dir)

directories = {'source_path': args.source_path, 
				'save_dir': args.save_dir,
				'patch_save_dir': patch_save_dir, 
				'mask_save_dir' : mask_save_dir, 
				'stitch_save_dir': stitch_save_dir} 

for key, val in directories.items():
	print("{} : {}".format(key, val))
	if key not in ['source_path']:
		os.makedirs(val, exist_ok=True)

seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				'keep_ids': 'none', 'exclude_ids': 'none'}
filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
vis_params = {'vis_level': -1, 'line_thickness': 250}
patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

if args.preset:
	preset_df = pd.read_csv(os.path.join('presets', args.preset))
	for key in seg_params.keys():
		seg_params[key] = preset_df.loc[0, key]

	for key in filter_params.keys():
		filter_params[key] = preset_df.loc[0, key]

	for key in vis_params.keys():
		vis_params[key] = preset_df.loc[0, key]

	for key in patch_params.keys():
		patch_params[key] = preset_df.loc[0, key]

parameters = {  'seg_params': seg_params,
				'filter_params': filter_params,
				'patch_params': patch_params,
				'vis_params': vis_params
				}

print(parameters)



seg_times, patch_times = seg_and_patch(**directories, **parameters,
										region_list = args.region,
										data_flag = args.data,
										patch_size = args.patch_size, step_size=args.step_size, 
										seg = args.seg,  use_default_params=False, save_mask = True, 
										stitch= args.stitch,
										patch_level=args.patch_level, patch = args.patch,
										process_list = process_list, auto_skip=args.no_auto_skip,
										expect_mag = args.mag
										)
