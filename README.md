# Wyniki: Mask R-CNN R50-FPN v2 vs DeepLabV3-ResNet50

Zbiór walidacyjny: 4 zdjęcia.

## Metryki pikselowe

| Model                 | IoU    | Precision | Recall | F1     |
|-----------------------|--------|-----------|--------|--------|
| Mask R-CNN R50-FPN v2 | 0.6710 | 0.8025    | 0.8037 | 0.8031 |
| DeepLabV3-ResNet50    | 0.8325 | 0.8842    | 0.9344 | 0.9086 |

| Model                 | TP      | FP     | FN     |
|-----------------------|---------|--------|--------|
| Mask R-CNN R50-FPN v2 | 97 373  | 23 961 | 23 782 |
| DeepLabV3-ResNet50    | 113 206 | 14 824 | 7 949  |

Mask R-CNN: próg pewności 0.5, próg maski 0.5.

## Metryki instancyjne (Mask AP)

| Model                 | Mask mAP | Mask AP50 | Mask AP75 |
|-----------------------|----------|-----------|-----------|
| Mask R-CNN R50-FPN v2 | 0.2652   | 0.4829    | 0.2924    |
| DeepLabV3-ResNet50*   | 0.0111   | 0.0238    | 0.0132    |

\*DeepLabV3 nie zwraca instancji. Instancje to składowe spójne przewidzianej maski, a score to średnie prawdopodobieństwo klasy, więc wynik jest przybliżeniem i nie jest porównywalny z Mask R-CNN.

## Wyniki dla poszczególnych zdjęć (IoU / F1)

| Zdjęcie                                      | Mask R-CNN      | DeepLabV3       |
|----------------------------------------------|-----------------|-----------------|
| e617ff0a6aa6040f9f32e79cfee59bfee76701a0.jpg | 0.8584 / 0.9238 | 0.8683 / 0.9295 |
| 74eb996790ceb5b7fd22ec8366532d78fef178b2.jpg | 0.5383 / 0.6998 | 0.8752 / 0.9335 |
| a88d571026ce81e54dfc6690f7d9eaa4e3cabcd4.jpg | 0.7928 / 0.8844 | 0.8365 / 0.9110 |
| 6c7eb452e66fe169f41178a6539fcabe9d498502.jpg | 0.5305 / 0.6932 | 0.7314 / 0.8449 |

## Wpływ progu pewności (Mask R-CNN)

| Próg | Detekcje | IoU    | Precision | Recall | F1     |
|------|----------|--------|-----------|--------|--------|
| 0.30 | 265      | 0.6743 | 0.7048    | 0.9397 | 0.8055 |
| 0.50 | 71       | 0.6710 | 0.8025    | 0.8037 | 0.8031 |
| 0.70 | 16       | 0.3795 | 0.8571    | 0.4052 | 0.5502 |
| 0.90 | 0        | 0.0000 | 0.0000    | 0.0000 | 0.0000 |
