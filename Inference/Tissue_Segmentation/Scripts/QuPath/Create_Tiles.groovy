import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)
mkdirs(pathOutput)

// Convert to downsample
double downsample = 4


// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
   
    .tileSize(512)              // Define size of each tile, in pixels
    .overlap(340)                // Define overlap, in pixel units at the export resolution
    
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'