# active_learning

## Датасет

CIFAR-10

## Алгоритмы


| Студент | Группа | Метод | Описание метода | Ссылка на статью |
| --- | --- | --- | --- | --- |
| Гаев Роман | 209М |  Expected Gradient Length (EGL) | Выбор примеров, которые имеют наибольшее влияние на градиент функции потерь. |  https://openreview.net/pdf?id=ryghZJBKPS |
|            |      |  BALD (Bayesian Active Learning by Disagreement) | Учитывает уменьшение энтропии при добавлении нового примера. | https://arxiv.org/pdf/1112.5745 |                        
| Лавренченко Мария | 209М |  Contrastive Active Learning (CAL)   |Выбирает такие контрастирующие между собой примеры, которые наилучшим образом помогают улучшить качество модели. | https://aclanthology.org/2021.emnlp-main.51.pdf|
| Панцырный Иван | 214М | Least Confidence (LC) | Выбор данных с наименьшей уверенностью модели в предсказании. | https://arxiv.org/pdf/cmp-lg/9407020 |
| Орачёв Алексей | 209М | Maximum Normalized Log-Probability (MNLP) | Использование нормализованных логарифмов вероятностей для выбора примеров. | https://aclanthology.org/W17-2630.pdf |
## Результаты
![img1](res/final.png)
Протестировал на MNIST, модель очень быстро сошлась к 1, CIFAR10 видимо достаточно сложен для выбранной
## Результаты (Старое)

| Алгоритм  | F1 при 0.01                | F1 при 0.1                | F1 при 0.2                |
|-----------|---------------------------|---------------------------|---------------------------|
| No AL | ![img1](res/no_al_0.01.png) | ![img1](res/no_al_0.1.png)  | ![img1](res/no_al_0.2.png)  |
| EGL | ![img1](res/egl_f1_0.01.png) | ![img2](res/egl_f1_0.1.png) | ![img3](res/egl_f1_0.2.png) |
| BALD | ![img1](res/bald_f1_0.01.png) | ![img2](res/bald_f1_0.1.png) | ![img3](res/bald_f1_0.2.png) |
| Contr |  ![img2](res/contr_f1_0.01.png) | ![img2](res/contr_f1_0.1.png) |  ![img2](res/contr_f1_0.2.png) |
| LC |  ![img2](res/ls_f1_0.01.png) |  ![img2](res/ls_f1_0.1.png)  |  ![img2](res/ls_f1_0.2.png) |
| MNLP | ![img2](res/mnlp_f1_0.01.png) | ![img2](res/mnlp_f1_0.1.png)  | ![img2](res/mnlp_f1_0.2.png)  |
