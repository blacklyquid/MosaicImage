from PIL import Image, ImageOps
import os, argparse, sys, random, math, cv2, numpy as np, time
import operator, jsonpickle
from pillow_heif import register_heif_opener
from pathlib import Path
from PIL.ExifTags import TAGS
register_heif_opener()

def print2d(d):
	for row in d:
		print(", ".join(map(str, row)))
# python3 mosaic.py -i /pool/IMG20251104081323.jpg -p /pool -o /output/mosaic.jpg

# Bit is a pixel
class Bit:
	def __init__(self, data: tuple[int,int,int]):
		self.data = data
		
	def sub(self, other) -> int:
		diff = 0
		for i in range(0,3):
			diff += abs(self.data[i] - other.data[i])
		return diff
			

	def __repr__(self):
		return f"[{self.data}]"



# image tile object. stores the position of the tile.
# Tile may be an entire image or just part of a larger one
# pos row and col of the tile
# Tiles are all 2x2
# data stores pixel colors for earch of the 4 pixels
class MosaicTile:
	def __init__(self, img: Image=None, pos:tuple[int,int]=(0,0), imagepath:str=None, size: int=3):
		
		self.pos = pos
		x, y = pos[0] * size, pos[1] * size

		if img is not None and isinstance(img, Image.Image):
		
			self.data = [[[0 for _ in range(3)] for _ in range(size)] for _ in range(size)]
			#self.data = [[[0] for _ in range(size)] for _ in range(size)]
			for r in range(0,size):
				for c in range(0,size):
					self.data[r][c] = Bit(img.getpixel(( x+r, y+c )))

		#self.exif(img)
		self.size = size
		self.x, self.y = (x, y)
		self.rotation = -1
		self.imagepath = imagepath
		self.score = 0
		self.used = 0

	def clone(self):
		cpy = MosaicTile(pos=self.pos, imagepath=self.imagepath)
		cpy.data = self.data
		cpy.score = self.score
		cpy.rotation = self.rotation

		return cpy


	def exif(self, img: Image):

		exif_data_raw = img.getexif()
		labeled_exif_data = {}

		if exif_data_raw:
			for tag_id, value in exif_data_raw.items():
				tag_name = TAGS.get(tag_id, tag_id)
				labeled_exif_data[tag_name] = value
		self.exif = labeled_exif_data


	# Given a target tile, find the best score/rotation
	def fitTarget(self, target, rotation:bool=True):
		
		diff = 0
		self.rotation = -1
		#self.score = target.tileDifference(self)
		self.score = self.sub(target)
		if rotation is True:

			score90 = self.rotate().sub(target)
			score180 = self.rotate(180).sub(target)
			score270 = self.rotate(270).sub(target)
			if score90 < self.score:
				self.score = score90
				self.rotation = Image.Transpose.ROTATE_90

			if score180 < self.score:
				self.score = score180
				self.rotation = Image.Transpose.ROTATE_180

			if score270 < self.score:
				self.score = score270
				self.rotation = Image.Transpose.ROTATE_270
	
	def sub(self, tile) -> int:
		if not isinstance(tile, MosaicTile):
			raise TypeError("Unsupported operand type(s) for -")

		diff = 0
		for r in range(0,self.size):
			for c in range(0,self.size):			
				diff += self.data[r][c].sub(tile.data[r][c])
		return diff

	def rotate(self, deg: int=90):
		# copy, rotate and return clone
		cpy = self.clone()
		cpy.data = [list(row) for row in zip(*cpy.data[::-1])]
		if deg >= 180:
			cpy.data = [list(row) for row in zip(*cpy.data[::-1])]
		if deg >= 270:
			cpy.data = [list(row) for row in zip(*cpy.data[::-1])]
		return cpy




	def __str__(self) -> str:
		rtrn = ""
		for d in self.data:
			rtrn = rtrn + f"{d}"
		return f"{self.imagepath}|Rot:{self.rotation}|used:{self.used}|{rtrn}"

class MosaicPreview:
	def __init__(self, active:bool=False):
		self.active = active

	def show(self, mosaic):
		if self.active is True:
			opencv_image = cv2.cvtColor(np.array(mosaic), cv2.COLOR_RGB2BGR)
			cv2.imshow("window", opencv_image)
			k = cv2.waitKey(1) & 0xFF
class Mosaic:

	def __init__(self, imagepath: str=None, resolution=25, filename="mosaic.jpg", repeat:int=0, rotation:bool=True):
		# is imagepath a real file
		self.imagepath = imagepath
		self.resolution = resolution
		self.output_filepath = filename
		self.mult = 1
		self.tiles = []
		self.pool = []
		self.mosaic = []
		self.rows = None
		self.cols = None
		self.repeat = repeat
		self.rotation = rotation
		self.BITS = 10
		self.use_cache = False
		self.preview = MosaicPreview()

	@property
	def bits(self) -> int:
		return self.BITS
	@bits.setter
	def bits(self, value:int):
		self.BITS = value

	@property
	def use_cache(self) -> bool:
		return self._use_cache
	@use_cache.setter
	def use_cache(self, value:bool):
		self._use_cache = value

	@property
	def rotation(self) -> bool:
		return self._rotation
	@rotation.setter
	def rotation(self, value:bool):
		self._rotation = value

	@property
	def repeat(self):
		return self._repeat
	@repeat.setter
	def repeat(self, value:int):
		self._repeat = value

	@property
	def show_build(self):
		return self.preview.active

	@show_build.setter
	def show_build(self, value:bool):
		if isinstance(value, (bool)):
			self.preview.active = value

	def setMosaic(self, imagepath: str):
		if not os.path.isfile(imagepath):
				raise Exception(f"The image '{imagepath}' could not be found")
		self.tiles = []
		img = Image.open(imagepath)
		#print(img.size)

		# resize image to fit even number of subimages of size resolution X resolution
		# but twice as big in each direction. each tile is 2px x 2px
		# each sample tile is going to be 2px x 2px
		(width, height) = (img.width // self.resolution * self.BITS, img.height // self.resolution * self.BITS)
		img = img.resize((width, height))
		#print(img.size)

		# width and height are twice as big as it should be
		self.cols = width // self.BITS
		self.rows = height // self.BITS

		for i in range(0, self.rows):
			for j in range(0, self.cols):
				self.tiles.append(MosaicTile(img, (j, i), size=self.BITS ))


	# find the optimal tile image for the sample area
	def findOptimalTile(self, target_tile: MosaicTile):

		for pool_tile in self.pool:
			pool_tile.fitTarget(target_tile, self.rotation)

		self.pool = sorted(self.pool, key=lambda tile: tile.score)
		if self.repeat < 1:
			return self.pool[0]

		for pool_tile in self.pool:
			if pool_tile.used < self.repeat:
				pool_tile.used += 1
				return pool_tile

	# build the mosaic image
	def build(self):

		# check how many images are in the pool
		if len(self.pool) < 1:
			raise Exception(f"There should be more than 0 images in the pool.")

		self.setMosaic(self.imagepath)

		mosaic = Image.new(mode="RGB", size=((self.resolution * self.cols)*self.mult, (self.resolution * self.rows)*self.mult))
		tile_counter = 0
		number_tiles = len(self.tiles)
		pool_size = len(self.pool)
		progress = MosaicProgressBar(number_tiles)
		print(f"Building the mosaic image with {number_tiles} tiles from {pool_size} images.")
		for target_tile in self.tiles:
			tile_counter += 1
			besttile = self.findOptimalTile(target_tile)

			img = Image.open(besttile.imagepath).convert('RGB')
			img = ImageOps.fit(img, (self.resolution*self.mult, self.resolution*self.mult), Image.Resampling.LANCZOS)

			# rotate the tile for best alignment
			if besttile.rotation > 0:
				img = img.transpose(besttile.rotation)

			mosaic.paste(img, (target_tile.pos[0]*self.resolution*self.mult, target_tile.pos[1]*self.resolution*self.mult))

			self.preview.show(mosaic)
			progress.advance().display()


		mosaic.save(self.output_filepath, format='jpeg')
		print(f"Saved the new mosaic image at {self.output_filepath}.")

	def getTile(self, imagepath:str):
		try:
			if not os.path.isfile(imagepath):
				raise Exception(f"The image '{imagepath}' could not be found")

			img = Image.open(imagepath)
			img = ImageOps.fit(img, (self.resolution, self.resolution), Image.Resampling.LANCZOS)
			img = img.resize((self.BITS, self.BITS))
			return MosaicTile(img, (0,0), imagepath=imagepath, size=self.BITS )
		except Exception as e:
			print("Image/Tile Error: ", e)
			return False

	def addDirectory(self, dirpath:str, recursive:bool=False):
		# check if dirpath is a directory
		if not os.path.isdir(dirpath):
			raise Exception(f"The image pool directory '{dirpath}' does not exist")

		pool = MosaicPool(dirpath, use_cache=self.use_cache, resolution=self.resolution, bits=self.BITS)
		self.pool = self.pool + pool.tiles
		pool.save()

		# if recursive adding of directories is turned on check for directories and add them
		if recursive is True:
			for sub in pool.subdirs:
				self.addDirectory(os.path.join(dirpath, sub))

	def __str__(self) -> str:
		rtrn = ""
		for p in self.pool:
			rtrn = rtrn + f"{p}"
		for t in self.tiles:
			rtrn = rtrn + f"{t}"
		return rtrn

class MosaicPool:
	def __init__(self, dirpath:str, resolution:int, use_cache:bool=False, bits=3):
		self.dirpath = dirpath
		self.resolution = resolution
		self.BITS = bits
		self.files = os.listdir(dirpath)
		self.cache = MosaicCache(dirpath, enabled=use_cache)
		self.use_cache = use_cache
		self.tiles = self.cache.contents
		self.image_types = ('jpg','HEIC')
		print(f"Loading {self.dirpath} pool")
		if not self.cache.loaded:
			self.load()
			self.cache.contents = self.tiles
		self.subdirs = [d for d in self.files if os.path.isdir(os.path.join(self.dirpath, d))]

	def load(self):
		self.imagefiles = [imgs for imgs in self.files if imgs.endswith(self.image_types)]
		progress = MosaicProgressBar(len(self.imagefiles))
		self.tiles = []
		for filename in self.imagefiles:
			tile = self.genTile(os.path.join(self.dirpath, filename))
			if tile:
				self.tiles.append( tile )
				progress.advance().display()

	def save(self):
		self.cache.save()

	def genTile(self, imagepath:str):
		if not os.path.isfile(imagepath):
			raise Exception(f"The image '{imagepath}' could not be found")
		try:
			img = Image.open(imagepath)
			img = ImageOps.fit(img, (self.resolution, self.resolution), Image.Resampling.LANCZOS)
			img = img.resize((self.BITS, self.BITS))
			return MosaicTile(img, (0,0), imagepath=imagepath, size=self.BITS)
		except Exception as e:
			print("Image/Tile Error: ", e)
			return False

class MosaicProgressBar:
	def __init__(self, total:int, start:int=0):
		self.prefix = ''
		self.suffix = ''
		self.fill = '█'
		self.print_end = "\r"
		self.current = 0
		self.total = total

	def advance(self):
		self.current += 1
		return self

	def display(self):
		length, rows = os.get_terminal_size(0)
		length -= 12
		percent = ("{0:.2f}").format(100 * (self.current / float(self.total)))
		filledLength = int(length * self.current // self.total)
		bar = self.fill * filledLength + '-' * (length - filledLength)
		print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end = self.print_end)
		# Print New Line on Complete
		if self.current == self.total:
			print()

class MosaicCache:
	def __init__(self, dirpath:str, enabled:bool=True, base:str="cache", end:str="_cache.json"):
		self.base_dir = base
		self.filename_end = end
		self.contents = False
		self.enabled = enabled
		self.filepath = self.genCacheFilename(dirpath)
		self.loaded = False
		if not os.path.isdir(base) and enabled is True:

			print(f"Creating directory '{base}'")
			try:
				Path(base).mkdir()
			except Exception as e:
				print("An error occured while trying to create the directory {base}.")
				self.base_dir = None
		if enabled is True:
			self.load()



	def genCacheFilename(self, dirpath):
		filename = dirpath.replace("/","")
		filename = filename.replace(" ","_")
		if self.base_dir is not None:
			filename = os.path.join(self.base_dir, filename)
		return f"{filename}{self.filename_end}"

	def load(self):
		self.loaded = False
		if os.path.isfile(self.filepath) and self.enabled is True:
			try:
				with open(self.filepath, 'r') as json_file:
					self.contents = jsonpickle.decode(json_file.read())
				cache_len = len(self.contents)
				print(f"Loaded {cache_len} from {self.filepath} cache")
				self.loaded = True
			except Exception as e:
				print(f"Failed loading {self.filepath} cache")
				self.loaded = False
		return self.loaded

	def save(self):
		if self.enabled is True:
			with open(self.filepath, 'w') as json_file:
		    		json_file.write(jsonpickle.encode(self.contents))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='mosaic', description="Create a mosaic image from a pool of pictures.")
	parser.add_argument("-d", "--display", action="store_true", help="Display the image as it is built. May error if using on a headless system.")
	parser.add_argument("-e", "--recursive", action="store_true", help="Recursively add directories to the pool.")
	parser.add_argument("-c", "--use_cache", action="store_true", help="Save pool image information in a file cache. mosaic_pool_cache.json")
	parser.add_argument("-n", "--no_rotation", action="store_true", help="Do not rotate tiles for best fit. Rotating takes time. Defualt is to rotate.")
	parser.add_argument("-i", "--image", required=True, type=str, help="Input file. The base image for the mosaic.")
	parser.add_argument("-p", "--pool", nargs='+', type=str, required=True, help="The directory containing a pool of pictures to use to create the mosaic.")
	parser.add_argument("-o", "--output", default="mosaic.jpg", type=str, help="Default 'mosaic.jpg'. Ouput file. The name of the file to save the mosaic as.")
	parser.add_argument("-m", "--multiplier", type=int, default=1, help="Increase the size of the output image by this amount.")
	parser.add_argument("-R", "--repeat", type=int, default=0, help="Allow thumb images to repeat this many times. Less than 1 is off. Default is 0.")
	parser.add_argument("-r", "--resolution", type=int, default=25, help="The size of the thumbnail in the mosaic. Defaults to 25.")
	parser.add_argument("-b", "--bits", type=int, default=3, help="The number of pixels to compare. Defaults to 3 = 3x3")


	args = parser.parse_args()

	# check input image
	if os.path.isfile(args.image) == False:
		raise Exception(f"The image '{args.image}' could not be found")
	# check image pool directories
	for dirpath in args.pool:
		if os.path.isdir(dirpath) == False:
			raise Exception(f"The image pool directory '{dirpath}' does not exist")

	if args.use_cache is True:
		print("Using cache files if possible")
		if not os.path.isdir("cache"):
			print("Creating directory 'cache'")
			Path("cache").mkdir()

	mosaic = Mosaic(args.image, resolution=args.resolution)
	mosaic.mult = args.multiplier
	mosaic.repeat = args.repeat
	mosaic.show_build = args.display
	mosaic.use_cache = args.use_cache
	mosaic.rotation = not args.no_rotation
	mosaic.bits = args.bits

	for directory in args.pool:
		try:
			mosaic.addDirectory(directory, recursive=args.recursive)
		except Exception as e:
			print(f"Adding directory {directory} failed: ", e)

	mosaic.build()

