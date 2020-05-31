# CoralClef2020
Object detection in an Image using Neural network

## Mask R-CNN
original repository - [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- Mask_RCNN_evaluate_coral
- Mask_RCNN_inference_coral
- Mask_RCNN_test_coral
- MRCNN_training_coral
## SSD
original repository - [https://github.com/pierluigiferrari/ssd_keras](https://github.com/pierluigiferrari/ssd_keras)
- ssd512_evaluation_coral
- ssd512_inference_coral
- ssd512_test_coral
- ssd512_training_coral
- weight_sampling_coral

## mAP
original repository - [https://github.com/Cartucho/mAP](https://github.com/Cartucho/mAP)


## ostatni
- flip_img - Některé obrázky v datasetu jsou vzhledem k anotacím otočené o 180° a je potřeba je otočit.
- save_to_json - Převedení poskytnutých anotací do formátu MS COCO.
- draw_bb - Ilustrační zobrazení rámečků v obrázcích.
- augmentations - Vygeneruje augmentované obrázky a soubor s novými anotacemi z původního datasetu a anotací vytvořených pomocí [save_to_json.py](https://github.com/strakaj/CoralClef2020/blob/master/ostatni/save_to_json.py). 
- split_dataset - Rozdělí dataset na validační a trénovací množinu v zadaném poměru. Obrázky jsou rozděleny podle instancí tříd, které obsahují, tak aby se v každé množině vyskytovali v zadaném poměru. To může způsobit, že výsledný poměr bude vyšší než zadaný. První část rozdělí neaugmentovaná data, druhá část pak přiřadí augmentované obrázky do množiny ve které je originální obrázek. Vstupem je soubor s anotacemi ve formátu MS COCO, případně soubor s augmentovanými anotacemi, výstupem je soubor s anotacemi pro validační a soubor s anotacemi pro trénovací množinu.


