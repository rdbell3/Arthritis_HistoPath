// Script written for QuPath v0.2.3
// Minimal working script to import labelled images
// (from the TileExporter) back into QuPath as annotations.

import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import static qupath.lib.gui.scripting.QPEx.*
import ij.IJ
import ij.process.ColorProcessor
import qupath.imagej.processing.RoiLabeling
import qupath.imagej.tools.IJTools
import java.util.regex.Matcher
import java.util.regex.Pattern


def imageName = ServerTools.getDisplayableImageName(getCurrentServer())

imageName = imageName[0..-1]

def directoryPath = "D:/Richard Data/Kiana_Homaira/MTPs for Richard/QP_MTPs/Majority_Vote_Results/" + imageName


//def directoryPath = 'R:/MK Anti-TNF 2nd Batch/Results_Majority_Vote/916 R K LVL2' // TO CHANGE
File folder = new File(directoryPath);
File[] listOfFiles = folder.listFiles();

listOfFiles.each { file ->
    def path = file.getPath()
    def imp = IJ.openImage(path)

    // Only process the labelled images, not the originals
    if (!path.endsWith(".tif"))
        return

    print "Now processing: " + path

    // Parse filename to understand where the tile was located
    def parsing = parseFilename(GeneralTools.getNameWithoutExtension(path))
    def classification = parsing[0]
    def x = parsing[1] as double
    def y = parsing[2] as double

    double downsample = 4 // TO CHANGE (if needed)
    ImagePlane plane = ImagePlane.getDefaultPlane()


    // Convert labels to ImageJ ROIs
    def ip = imp.getProcessor()
    if (ip instanceof ColorProcessor) {
        throw new IllegalArgumentException("RGB images are not supported!")
    }

    int n = imp.getStatistics().max as int
    if (n == 0) {
        print 'No objects found!'
        return
    }
    def roisIJ = RoiLabeling.labelsToConnectedROIs(ip, n)



    // Convert ImageJ ROIs to QuPath ROIs
    def rois = roisIJ.collect {
        if (it == null)
            return
        return IJTools.convertToROI(it, -x/downsample, -y/downsample, downsample, plane);
    }

    // Remove all null values from list
    rois = rois.findAll{null != it}

    // Convert QuPath ROIs to objects
    def pathObjects = rois.collect {
        return PathObjects.createAnnotationObject(it, getPathClass(classification))
    }
    addObjects(pathObjects)
}

resolveHierarchy()

String[] parseFilename(String filename) {
    def p = Pattern.compile("40x (.*) \\[x=(.+?),y=(.+?),")
    parsed = []
    Matcher m = p.matcher(filename)
    if (!m.find())
        throw new IOException("Filename does not contain classification and/or tile position")
    
    parsed << m.group(1)
    parsed << (m.group(2) as double)
    parsed << (m.group(3) as double)

    return parsed
}

selectObjectsByClassification("Trabecular Bone");
mergeSelectedAnnotations()

selectObjectsByClassification("Cortical Bone");
mergeSelectedAnnotations()

selectObjectsByClassification("Synovium");
mergeSelectedAnnotations()

selectObjectsByClassification("Growth Plate");
mergeSelectedAnnotations()

selectObjectsByClassification("Muscle-Tendon");
mergeSelectedAnnotations()

selectObjectsByClassification("Fat");
mergeSelectedAnnotations()

selectObjectsByClassification("Bone Marrow");
mergeSelectedAnnotations()

selectObjectsByClassification("Meniscus-Cartilage");
mergeSelectedAnnotations()

selectObjectsByClassification("Artifact");
mergeSelectedAnnotations()

selectObjectsByClassification("Cartilage");
mergeSelectedAnnotations()

selectObjectsByClassification("Bone Marrow Fat");
mergeSelectedAnnotations()

selectObjectsByClassification("Meniscus");
mergeSelectedAnnotations()


