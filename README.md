# MPMABP
MPMABP: A CNN and Bi-LSTM-Based Method for Predicting Multi-Activities of Bioactive Peptides

**Abstract:** Bioactive peptides are typically small functional peptides with 2–20 amino acid residues
and play versatile roles in metabolic and biological processes. Bioactive peptides are multi-functional,
so it is vastly challenging to accurately detect all their functions simultaneously. We proposed a
convolution neural network (CNN) and bi-directional long short-term memory (Bi-LSTM)-based
deep learning method (called MPMABP) for recognizing multi-activities of bioactive peptides. The
MPMABP stacked five CNNs at different scales, and used the residual network to preserve the
information from loss. The empirical results showed that the MPMABP is superior to the state-ofthe-art methods. Analysis on the distribution of amino acids indicated that the lysine preferred to
appear in the anti-cancer peptide, the leucine in the anti-diabetic peptide, and the proline in the
anti-hypertensive peptide. The method and analysis are beneficial to recognize multi-activities of
bioactive peptides.

![image](https://github.com/Good-Ly/MFBPP/blob/main/figures/MFBPP.jpg)

## Dataset
![image](https://github.com/Good-Ly/MFBPP/blob/main/figures/dataset.jpg)


## installation
- python==3.7.11
- keras==2.2.4
- tensorflow ==1.13.1    
- tensorflow-gpu==1.13.1
- numpy==1.21.5

## MPMABP_model
https://drive.google.com/drive/folders/10oPJ_koydNV4Cw_BjpwLtiLq7CpaNlGN?usp=sharing

## Citation
Li, Y.; Li, X.; Liu, Y.; Yao, Y.; Huang, G. MPMABP: A CNN and Bi-LSTM-Based Method for Predicting Multi-Activities of Bioactive Peptides. Pharmaceuticals 2022, 15, 707. https://doi.org/10.3390/ph15060707
