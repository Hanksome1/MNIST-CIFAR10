# Initial setup
- optimizer: SDG
- learning rate: 0.5 
- Data Augmentation : nothing 
- dropout: 0.2
###### Result: 
```python
Epoch: 1  loss:  2.1876590613209075
Testing Loss :  2.1380900144577026
Accuracy : 22.68
Epoch: 2  loss:  2.034442976917452
Testing Loss :  1.9527679800987243
Accuracy : 33.4
Epoch: 3  loss:  1.8486753500940856
Testing Loss :  2.1285975217819213
Accuracy : 24.19
Epoch: 4  loss:  1.7194093689894128
Testing Loss :  1.8478568077087403
Accuracy : 38.88
Epoch: 5  loss:  1.6004509697179965
Testing Loss :  1.8955473065376283
Accuracy : 39.13
Epoch: 6  loss:  1.4745507446853705
Testing Loss :  1.864353322982788
Accuracy : 39.87
Epoch: 7  loss:  1.295832901659524
Testing Loss :  1.7992312788963318
Accuracy : 43.59
Epoch: 8  loss:  1.1193504985183707
Testing Loss :  1.719679880142212
Accuracy : 49.85
Epoch: 9  loss:  1.0146882866349671
Testing Loss :  1.7669610619544982
Accuracy : 50.99
Epoch: 10  loss:  0.9384255420292735
Testing Loss :  1.7825339555740356
Accuracy : 51.81
Epoch: 11  loss:  0.8859367564206233
Testing Loss :  1.8536918997764587
Accuracy : 51.4
Epoch: 12  loss:  0.8478901053938415
Testing Loss :  1.9049222707748412
Accuracy : 52.39
Epoch: 13  loss:  0.8200035733182717
Testing Loss :  1.901247239112854
Accuracy : 53.12
Epoch: 14  loss:  0.8004688365608835
Testing Loss :  1.9068991422653199
Accuracy : 53.28
Epoch: 15  loss:  0.7890174080191366
Testing Loss :  1.914801561832428
Accuracy : 53.17
Epoch: 16  loss:  0.7781119096233412
Testing Loss :  1.9409478545188903
Accuracy : 53.58
Epoch: 17  loss:  0.7721491816937162
Testing Loss :  2.0065648794174193
Accuracy : 53.23
Epoch: 18  loss:  0.7497742896723321
Testing Loss :  1.9182681441307068
Accuracy : 60.82
Epoch: 19  loss:  0.48923961580028313
Testing Loss :  1.5558918833732605
Accuracy : 66.87
Epoch: 20  loss:  0.4056342942902194
Testing Loss :  1.5879979491233827
Accuracy : 67.94
```
The final accuracy is 67%

# Data Augmentation
## 1. Normalization
###### Result:
```python 
Epoch: 1  loss:  2.222368495391153
Testing Loss :  2.2263991832733154
Accuracy : 17.93
Epoch: 2  loss:  2.0985749841041272
Testing Loss :  2.2745980978012086
Accuracy : 12.79
Epoch: 3  loss:  1.9399079053908053
Testing Loss :  1.9850092768669128
Accuracy : 27.84
Epoch: 4  loss:  1.670207411279459
Testing Loss :  1.926323449611664
Accuracy : 36.07
Epoch: 5  loss:  1.4001512387219597
Testing Loss :  1.6944506168365479
Accuracy : 46.4
Epoch: 6  loss:  1.056688682807376
Testing Loss :  1.4289908766746522
Accuracy : 55.98
Epoch: 7  loss:  0.7922757652199939
Testing Loss :  1.1861487746238708
Accuracy : 67.18
Epoch: 8  loss:  0.4981214925122764
Testing Loss :  1.209059226512909
Accuracy : 68.07
Epoch: 9  loss:  0.3571502172895481
Testing Loss :  1.3313060522079467
Accuracy : 68.51
Epoch: 10  loss:  0.2587197319654476
Testing Loss :  1.4354074716567993
Accuracy : 68.77
Epoch: 11  loss:  0.19180141638055126
Testing Loss :  1.4938153266906737
Accuracy : 68.98
Epoch: 12  loss:  0.14413053884714022
Testing Loss :  1.5226081728935241
Accuracy : 70.0
Epoch: 13  loss:  0.10662550641500089
Testing Loss :  1.6822609186172486
Accuracy : 70.07
Epoch: 14  loss:  0.08577047820999305
Testing Loss :  1.626682984828949
Accuracy : 70.58
Epoch: 15  loss:  0.06945651797539312
Testing Loss :  1.666055405139923
Accuracy : 70.94
Epoch: 16  loss:  0.052299437727505976
Testing Loss :  1.6610400438308717
Accuracy : 71.34
Epoch: 17  loss:  0.04049610058303111
Testing Loss :  1.668238115310669
Accuracy : 71.31
Epoch: 18  loss:  0.03668223513885112
Testing Loss :  1.6927425861358643
Accuracy : 71.69
Epoch: 19  loss:  0.032950261149831496
Testing Loss :  1.7294896602630616
Accuracy : 72.21
Epoch: 20  loss:  0.029040740799270836
Testing Loss :  1.7673702359199523
Accuracy : 72.21

```

###### loss:
![[Pasted image 20240515085606.png]]
Since the loss is still decreasing, set higher epoch. 


### Epoch: 30
###### Result: 
```python 
Epoch: 1  loss:  1.6079284227107797
Testing Loss :  1.6913674473762512
Accuracy : 39.34
Epoch: 2  loss:  1.2416760278937151
Testing Loss :  1.3363384008407593
Accuracy : 51.46
Epoch: 3  loss:  1.083277809269288
Testing Loss :  1.0620853900909424
Accuracy : 62.67
Epoch: 4  loss:  0.9819082983619417
Testing Loss :  0.9184267580509186
Accuracy : 68.47
Epoch: 5  loss:  0.9000483977672694
Testing Loss :  0.8985731422901153
Accuracy : 68.5
Epoch: 6  loss:  0.8358923622866725
Testing Loss :  0.7858246326446533
Accuracy : 72.82
Epoch: 7  loss:  0.7775799441139412
Testing Loss :  0.7692176282405854
Accuracy : 73.27
Epoch: 8  loss:  0.7358449349165572
Testing Loss :  0.7245618522167205
Accuracy : 75.59
Epoch: 9  loss:  0.6965684061465056
Testing Loss :  0.715008544921875
Accuracy : 75.56
Epoch: 10  loss:  0.6516009426635244
Testing Loss :  0.7154451966285705
Accuracy : 75.36
Epoch: 11  loss:  0.6189272121319076
Testing Loss :  0.6994204342365264
Accuracy : 76.02
Epoch: 12  loss:  0.5874658936201154
Testing Loss :  0.6778360664844513
Accuracy : 76.77
Epoch: 13  loss:  0.5660796162798581
Testing Loss :  0.6588635861873626
Accuracy : 77.44
Epoch: 14  loss:  0.5434877171807582
Testing Loss :  0.6520662426948547
Accuracy : 77.6
Epoch: 15  loss:  0.5192754244065041
Testing Loss :  0.6525824010372162
Accuracy : 77.49
Epoch: 16  loss:  0.49930869555458085
Testing Loss :  0.6576515316963196
Accuracy : 77.44
Epoch: 17  loss:  0.48340073309819714
Testing Loss :  0.653002268075943
Accuracy : 77.49
Epoch: 18  loss:  0.4607844401503463
Testing Loss :  0.6468328714370728
Accuracy : 77.71
Epoch: 19  loss:  0.44547752460555345
Testing Loss :  0.6378010749816895
Accuracy : 77.87
Epoch: 20  loss:  0.4292183984880862
Testing Loss :  0.6416742444038391
Accuracy : 77.98
Epoch: 21  loss:  0.41720361182528076
Testing Loss :  0.6300802111625672
Accuracy : 78.33
Epoch: 22  loss:  0.41057537408436046
Testing Loss :  0.6311923325061798
Accuracy : 78.43
Epoch: 23  loss:  0.400481933265772
Testing Loss :  0.6272027313709259
Accuracy : 78.56
Epoch: 24  loss:  0.39049159400069805
Testing Loss :  0.6223631620407104
Accuracy : 78.59
Epoch: 25  loss:  0.38272886544161133
Testing Loss :  0.626961886882782
Accuracy : 78.56
Epoch: 26  loss:  0.37454912217948444
Testing Loss :  0.6226541340351105
Accuracy : 78.91
Epoch: 27  loss:  0.36346044264676625
Testing Loss :  0.6261791467666626
Accuracy : 78.72
Epoch: 28  loss:  0.35920859319741466
Testing Loss :  0.6248948514461518
Accuracy : 78.56
Epoch: 29  loss:  0.354569401077526
Testing Loss :  0.622955983877182
Accuracy : 78.8
Epoch: 30  loss:  0.34584103617102596
Testing Loss :  0.6201816856861114
Accuracy : 78.93

```
###### loss:
![[Pasted image 20240515090402.png]]
## 2. Rotation and Flip
[[Data Augmentation]]
![[Pasted image 20240525190814.png]]
accuracy: 71.2%

### Dropout: 0.9 
Accuracy: 73.99%
![[Pasted image 20240515210600.png]]


