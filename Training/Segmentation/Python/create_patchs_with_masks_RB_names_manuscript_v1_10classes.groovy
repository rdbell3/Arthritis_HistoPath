import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles', name)
mkdirs(pathOutput)

// Convert to downsample
double downsample = 4

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Synovium',1)
    //.addLabel('Muscle-Tendon', 2)      // Choose output labels (the order matters!)
    .addLabel('Muscle and Tendon', 2)
    .addLabel('Artifact',3)    
    .addLabel('Growth Plate',4)
    .addLabel('Bone Marrow',5)
    .addLabel('Cortical Bone',6)
    .addLabel('Trabecular Bone',7)
    .addLabel('Meniscus',8)
    //.addLabel('Cartilage-Meniscus',8)
    //.addLabel('Meniscus-Cartilage',8)
    .addLabel('Cartilage',9)
    .addLabel('Fat', 10)    
    //.addLabel('Bone Marrow Fat',11)    



    
    .multichannelOutput(true)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .labeledImageExtension('.tif') 
    .tileSize(512)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(340)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'
