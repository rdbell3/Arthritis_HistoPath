// Create whole slide annotation and select it
createSelectAllObject(true);
//run SLIC plugin
runPlugin('qupath.imagej.superpixels.SLICSuperpixelsPlugin', '{"sigmaMicrons": 5.0,  "spacingMicrons": 20.0,  "maxIterations": 1,  "regularization": 0.01,  "adaptRegularization": false,  "useDeconvolved": false}');

//select tiles from SLIC and run Calculate Features plugin
selectDetections();
runPlugin('qupath.lib.algorithms.IntensityFeaturesPlugin', '{"pixelSizeMicrons": 2.0,  "region": "ROI",  "tileSizeMicrons": 25.0,  "colorOD": true,  "colorStain1": true,  "colorStain2": true,  "colorStain3": true,  "colorRed": true,  "colorGreen": true,  "colorBlue": true,  "colorHue": true,  "colorSaturation": true,  "colorBrightness": true,  "doMean": true,  "doStdDev": true,  "doMinMax": true,  "doMedian": true,  "doHaralick": true,  "haralickDistance": 1,  "haralickBins": 32}');
addShapeMeasurements("AREA", "LENGTH", "CIRCULARITY", "SOLIDITY", "MAX_DIAMETER", "MIN_DIAMETER", "NUCLEUS_CELL_RATIO")
selectAnnotations();
runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons": 40.0,  "smoothWithinClasses": false}');
runPlugin('qupath.lib.plugins.objects.SmoothFeaturesPlugin', '{"fwhmMicrons": 80.0,  "smoothWithinClasses": false}');

// resolved hierarchy
resolveHierarchy()
// Find parent annotation (level == 0), aka annotation created in step 1
def firstAnnotation = getAnnotationObjects().findAll{it.getLevel() == 1}

// Remove it
removeObjects(firstAnnotation, true)