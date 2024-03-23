# Arthritis_HistoPath
This is a repository of the training and inference scripts used in Bell and Brendal et al, AMSCP of Inflammatory Arthritis 

In our paper, Automated multi-scale computational pathotyping (AMSCP) of inflamed synovial tissue, we train, validate and deploy in real world settinges both a tissue segmentation model and cell classification model in the context of mouse and human synovial inflammatory arthritis.  

![Segmentation_Example](https://github.com/rdbell3/Arthritis_HistoPath/assets/46380784/9dca2722-3fab-438d-a409-ade34bd38902)

Our ten tissue mouse segmentation model was trained and validated on both healthy and diseased inflammatory arthritis H&E sections (n=264). Within the training set we had two different models of inflammatory arthritis, the TNF transgenic mouse and the Zymosan induce arthritis model that were procces and stained at different times (Batch A vs Batch B). This diverse training set allowed for improved performance and we achieved a frequency weighted mean intersection over union of 0.94 ± 0.01.

![Cell Classification Example](https://github.com/rdbell3/Arthritis_HistoPath/assets/46380784/71e7ce55-1420-4700-9a7c-457f7d25eca5)

Our human synovial cell classification model was trained on 2,639 pathologist annotated cells group into 7 cell types and achieved an F1 score of 0.85 ± 0.01. Lymphocytes and plasma cells were some of these best performing classificationas and were validated with immunohistochemistry.

In this repository there are methods and scripts for how we trained our models and how to use our models for inference.


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Title:
Automated multi-scale computational pathotyping (AMSCP) of inflamed synovial tissue

Authors:
Richard D. Bell §, Matthew Brendel §, Maxwell Konnaris, Justin Xiang, Miguel Otero, Mark A. Fontana, Zilong Bai, Accelerating Medicines Partnership Rheumatoid Arthritis and Systemic Lupus Erythematosus (AMP RA/SLE) Consortium, Daria Krenitsky, Nida Meednu, Javier Rangel-Moreno, Dagmar Scheel-Toellner, Hayley Carr, Saba Nayar, Jack McMurray, Edward DiCarlo, Jennifer Anolik, Laura Donlin, Dana Orange, H. Mark Kenney, Andrew Filer†, Edward M. Schwarz†, Lionel B Ivashkiv† and Fei Wang†

§, † Authors contributed equally


Abstract:

Rheumatoid arthritis (RA) is a complex immune-mediated inflammatory disorder in which patients suffer from inflammatory-erosive arthritis.  Recent advances on histopathology heterogeneity of RA pannus tissue revealed three distinct phenotypes based on cellular composition (pauci-immune, diffuse and lymphoid), suggesting distinct etiologies that warrant specific targeted therapy.  Thus, cost-effective alternatives to clinical pathology phenotyping are needed for research and disparate healthcare.  To this end, we developed an automated multi-scale computational pathotyping (AMSCP) pipeline with two distinct components that can be leveraged together or independently: 1) segmentation of different tissue types to characterize tissue-level changes, and 2) cell type classification within each tissue compartment that assesses change across disease states. Initial training and validation were completed on 264 knee histology sections from mice with TNF-transgenic (n=233) and injected zymosan induced (n=32) inflammatory arthritis.  Peak tissue segmentation performance with a frequency weighted mean intersection over union was 0.94 ± 0.01 and peak cell classification F1 was 0.88 ± 0.03. We then leveraged these models and adapted them to analyze RA pannus tissue clinically phenotyped as pauci-immune (n=5), diffuse (n=28) and lymphoid (n=27), achieving peak cell classification performance with F1 score of 0.85 ± 0.01.  Regression analysis demonstrated a highly significant correlation between AMSCP of lymphocyte percent vs average Krenn Inflammation Score (rho = 0.88; p<0.0001), plasma cell counts vs immunofluorescent CD138+ cell counts (rho = 0.86; p<0.002), and lymphocyte counts vs immunofluorescent CD3+/CD20+ cell counts (rho = 0.97; p<0.0001). Importantly, we can distinguish a lymphoid case from a diffuse case with a simple threshold of 0.82% of plasma cells, demonstrating the phenotyping potential of our automated approach vs. a clinical pathologist with a ROC-AUC of 0.82 ± 0.06. Taken together, we find AMSCP to be a valuable cost-effective method for research.  Follow-up studies to assess its clinical utility are warranted.


**License and Terms of Use**

© Bell Lab. This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the Arthritis_HistoPath model and its derivatives, which include models trained on outputs from the Arthritis_HistoPath model or datasets created from the UNI model, is prohibited and requires prior approval. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the UNI model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model.


