# Arthritis_HistoPath
This is a repository of the training and inference scripts used in Bell and Brendal et al, AMSCP of Inflammatory Arthritis 

Title:
Automated multi-scale computational pathotyping (AMSCP) of inflamed synovial tissue

Authors:
Richard D. Bell1,2,*,§, Matthew Brendel2, §, Maxwell Konnaris1, Justin Xiang3, Miguel Otero1,2, Mark A. Fontana1,2, Zilong Bai2, Accelerating Medicines Partnership Rheumatoid Arthritis and Systemic Lupus Erythematosus (AMP RA/SLE) Consortium, Daria Krenitsky5, Nida Meednu5, Javier Rangel-Moreno5, Dagmar Scheel-Toellner4, Hayley Carr4, Saba Nayar4, Jack McMurray4, Edward DiCarlo1, Jennifer Anolik5, Laura Donlin1, Dana Orange7, H. Mark Kenney6, Andrew Filer4†, Edward M. Schwarz6†, Lionel B Ivashkiv1,2†  and Fei Wang2†
§, † Authors contributed equally

Affiliations:![300-0143_Diffuse_5x_Overlay](https://github.com/rdbell3/Arthritis_HistoPath/assets/46380784/e18bbfdf-c745-469e-bba2-f7d2fd5d5f44)

 1 Arthritis and Tissue Degeneration Program, Hospital for Special Surgery, New York, NY
 2 Department of Population Health Sciences, Weill Cornell Medical College, New York, NY
 3 Horace Greely High School, Chappaqua, NY
 4 Rheumatology Research Group, Institute for Inflammation and Ageing, University of Birmingham, NIHR Birmingham Biomedical Research Center and Clinical Research Facility, University of Birmingham, Queen Elizabeth Hospital, Birmingham, UK
 5 Allergery, Immunology and Rheumatology Division, University of Rochester Medical Center, Rochester, NY
 6 Center for Musculoskeletal Research, University of Rochester Medical Center, Rochester, NY
 7 The Rockefeller University, New York, NY


Abstract:
Rheumatoid arthritis (RA) is a complex immune-mediated inflammatory disorder in which patients suffer from inflammatory-erosive arthritis.  Recent advances on histopathology heterogeneity of RA pannus tissue revealed three distinct phenotypes based on cellular composition (pauci-immune, diffuse and lymphoid), suggesting distinct etiologies that warrant specific targeted therapy.  Thus, cost-effective alternatives to clinical pathology phenotyping are needed for research and disparate healthcare.  To this end, we developed an automated multi-scale computational pathotyping (AMSCP) pipeline with two distinct components that can be leveraged together or independently: 1) segmentation of different tissue types to characterize tissue-level changes, and 2) cell type classification within each tissue compartment that assesses change across disease states. Initial training and validation were completed on 264 knee histology sections from mice with TNF-transgenic (n=233) and injected zymosan induced (n=32) inflammatory arthritis.  Peak tissue segmentation performance with a frequency weighted mean intersection over union was 0.94 ± 0.01 and peak cell classification F1 was 0.88 ± 0.03. We then leveraged these models and adapted them to analyze RA pannus tissue clinically phenotyped as pauci-immune (n=5), diffuse (n=28) and lymphoid (n=27), achieving peak cell classification performance with F1 score of 0.85 ± 0.01.  Regression analysis demonstrated a highly significant correlation between AMSCP of lymphocyte percent vs average Krenn Inflammation Score (rho = 0.88; p<0.0001), plasma cell counts vs immunofluorescent CD138+ cell counts (rho = 0.86; p<0.002), and lymphocyte counts vs immunofluorescent CD3+/CD20+ cell counts (rho = 0.97; p<0.0001). Importantly, we can distinguish a lymphoid case from a diffuse case with a simple threshold of 0.82% of plasma cells, demonstrating the phenotyping potential of our automated approach vs. a clinical pathologist with a ROC-AUC of 0.82 ± 0.06. Taken together, we find AMSCP to be a valuable cost-effective method for research.  Follow-up studies to assess its clinical utility are warranted.

Repository Notes:

This repositry is organized into training and inference sections, with both tissue segmention and cell classification sections within them. In the training tissue segmentation section, these are QuPATh .groovy scripts describing how to create the SLIC superpixels and the extracted feauters used in the manuscript



