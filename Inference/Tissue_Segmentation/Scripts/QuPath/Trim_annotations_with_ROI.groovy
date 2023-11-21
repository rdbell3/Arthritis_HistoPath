import qupath.lib.roi.RoiTools

// Get newROI annotation object
def newAnno = getAnnotationObjects().findAll{it.getPathClass() == getPathClass("ROI")}[0]

// Get annotations that are overlapping with newROI annotation, excluding itself
def region = ImageRegion.createInstance(newAnno.getROI())
def overlapAnno = getCurrentHierarchy().getObjectsForRegion(null, region, null)
overlapAnno.remove(newAnno)

// Intersect newROI with each annotation that overlaps with it
def intersectAnno = overlapAnno.parallelStream().map({ anno ->
    def roi = anno.getROI()
    def intersectROI = RoiTools.intersection([newAnno.getROI(), roi])
    def intersectObj = PathObjects.createAnnotationObject(intersectROI, anno.getPathClass())
    return intersectObj
}).collect().toList()

clearAnnotations() // delete all annotations
addObjects(intersectAnno) // restore intersected annotations