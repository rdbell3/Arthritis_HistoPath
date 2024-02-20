run("Colors...", "foreground=white background=black selection=cyan");
run("Set Measurements...", "area mean standard modal min centroid center perimeter shape feret's integrated median skewness kurtosis area_fraction limit redirect=None decimal=2");
run("Clear Results");
roiManager("reset");
run("Close All");
//setBatchMode(true)

root_path = 'D:\\AMP\\All_40x_Not_Annotated\\Output_5\\'



Dialog.create("Directory Selection");
Dialog.addMessage("Please Select the Folder that Contains the Images for Mask Creation");
Dialog.show();

imagedirectory=getDirectory("Choose the Folder");
print(imagedirectory);
imagelist = getFileList(imagedirectory);
//images = Array.sort(imagelist);

for (NumImages = 0; NumImages<imagelist.length; NumImages++) {
	if (endsWith(imagelist[NumImages], '.png')) {
		
		open(imagedirectory+ imagelist[NumImages]);
		run("Set Scale...", "distance=0 known=0 unit=pixel");
		
		name = getTitle();
		getDimensions(width, height, channels, slices, frames);

		namenoext = File.nameWithoutExtension;
		run("Duplicate...", "title=FinImg");
		selectWindow(name);

		run("Colour Deconvolution", "vectors=[User values] [r1]=0.49015734 [g1]=0.76897085 [b1]=0.41040173 [r2]=0.04615336 [g2]=0.8420684 [b2]=0.5373925 [r3]=0.76328504 [g3]=0.001 [b3]=0.64606184");

		selectWindow("FinImg");

		roi_direc = root_path + '\\rois\\'+ namenoext + '/';
		roilist = getFileList(root_path + '\\rois\\' + namenoext + '/');
		roilist = Array.sort(roilist);

		Array.print(roilist);

//////////// Load ROIS
		
		for (num_rois = 0; num_rois < roilist.length; num_rois++) {
		roiManager("Open", roi_direc+ roilist[num_rois]);

		}
		
///////////// Measure Nuclear Hematoxylin
	selectWindow(name+ "-(Colour_1)");
	run("Grays");
	run("Invert");
	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	    }
		
	print(roiManager("count"));
	saveAs("Results", root_path + '\\results_excels\\' + namenoext + '_hemo-nucleus.txt');


	run("Clear Results");

//////////// Measure Nuclear Eosin

	selectWindow(name+ "-(Colour_2)");
	run("Grays");
	run("Invert");
	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	}
	print(roiManager("count"));
	saveAs("Results", root_path + '\\results_excels\\' + namenoext + '_eosin-nucleus.txt');


	run("Clear Results");

//////////// Measure Nuclear Residual

	selectWindow(name+ "-(Colour_3)");
	run("Grays");
	run("Invert");
	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	}
	print(roiManager("count"));
	saveAs("Results", root_path + '\\results_excels\\' + namenoext + '_residual-nucleus.txt');
	

	run("Clear Results");

///////// Block out nuclei and enloarg ROI to measure cytoplasm

	newImage("Subtract_Nuclei", "8-bit black", width, height, 1);
	selectWindow("Subtract_Nuclei");
		
	num_rois = roiManager('count');
	rois = roiManager("select", Array.getSequence(num_rois));
	roiManager("Fill");
	
	imageCalculator("Subtract create", name+ "-(Colour_1)", "Subtract_Nuclei");
	imageCalculator("Subtract create", name+ "-(Colour_2)", "Subtract_Nuclei");
	imageCalculator("Subtract create", name+ "-(Colour_3)", "Subtract_Nuclei");	
	
	roiManager("deselect");
	for (i = 0; i<roilist.length; i++) {
    	roiManager('select', i);
    	wait(10);
    	/// Enlarge 10 for 20x and 20 for 40x
    	run("Enlarge...", "enlarge=20");
    	roiManager("Update");
    	wait(10);
    	roiManager("select", i);
	}

////////// Measure Cytoplasim Hematoxyin
	
	selectWindow("Result of "+name+ "-(Colour_1)");
	setThreshold(15,255);

	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	}
	print(roiManager("count"));
	saveAs("Results", root_path + "\\results_excels\\" + namenoext + "_hemo-cyto.txt");


	run("Clear Results");

//////////// Measure Cytoplasmic Eosin

	selectWindow("Result of "+name+ "-(Colour_2)");
	setThreshold(15,255);

	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	}
	print(roiManager("count"));
	saveAs("Results", root_path + "\\results_excels\\" + namenoext + "_eosin-cyto.txt");


	run("Clear Results");

///////////// Measure Cytoplasmic Residual

	selectWindow("Result of "+name+ "-(Colour_3)");
	setThreshold(15,255);

	roiManager("deselect");
	roiManager("measure");

	IJ.renameResults("Results"); // otherwise below does not work...
	for (row=0; row<nResults; row++) {
		sum = roilist[row];
	    setResult("Name", row, sum);
	}
	print(roiManager("count"));
	saveAs("Results", root_path + "\\results_excels\\" + namenoext + "_residual-cyto.txt");


	run("Clear Results");


	roiManager("reset");
	run("Close All");
}

}
	