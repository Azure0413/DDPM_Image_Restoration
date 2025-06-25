## Diffusion-Based Inverse Compression: Restoring High-Fidelity Images from Compressed Visuals

### Introduction  
本研究主要透過DDPM可以學習Mixture of Gaussian為出發點，透過自行設計的solver去讓模型學習score function的能力提升，同時根據破壞式壓縮的特性，設計color_loss_function去改善學習的效果。  
本研究重要改善之痛點為  
1.	破壞式壓縮可能伴隨artifacts，而artifacts在做後續下游任務或是人類肉眼觀感上會有重大影響。像是圖像分類、語意切割任務上，很多因為artifacts或圖像品質不穩導致有錯誤的分析，另外artifacts的存在也會影響人類的觀感。  
2.	破壞式壓縮為非線性的影像修復任務，常見的paper在做圖像修復都從線性degraded matrix來做，頂多一兩項非線性任務的修復，如：DDRM。因此，我希望可以讓整體圖像在破壞後可以有好的修復結果，除了達到PIR（Perceptual Image Restoration）外，還可以達到TIR（Task-oriented Image Restoration）。  

### Method  
1.	預計使用diffusion model為主體來完成此任務，因為diffusion model其實他不是單純做加噪和去噪的動作，它理論的意義其實是在過程中學習如何將圖片回復到前一個狀態，這種特性跟image restoration非常相似，目前諸多論文都採用diffusion model來實作，有別於過去的VAE和GAN，Diffusion的概念更貼近image restoration的性質。  
2.	那我實際要做的事是將最主流的破壞式壓縮演算法jpeg（最為常見）、webp（網頁很常用）、avif（效果較前兩者好，使用率漸增），因為這三者都是在量化過程中採用DCT（Discrete Cosine Transform），所以可能較多共通性。  

### Results  
![image](https://github.com/Azure0413/DDPM_Image_Restoration/blob/main/src/jpeg_result.png)
![image](https://github.com/Azure0413/DDPM_Image_Restoration/blob/main/src/webp_result.png)
![image](https://github.com/Azure0413/DDPM_Image_Restoration/blob/main/src/avif_result.png)

### Experiments
![image](https://github.com/Azure0413/DDPM_Image_Restoration/blob/main/src/performance_summary.png)
![image](https://github.com/Azure0413/DDPM_Image_Restoration/blob/main/src/avif.png)
