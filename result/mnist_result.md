PgThis is the Experiment Result of [[MNIST|MNIST]]
# 1. Default
the default setting: 
- Learning rate : 0.0001
- Optimizer : `SGD`
- Momentum : no value
- Gamma: 0.7
###### Result : 
```bash
(ML) hank@Hanksome:~/ML/hw_mnist$ python3 main.py 
Epoch: 1  loss:  2.2612390518188477
Accuracy : 23.48
Epoch: 2  loss:  2.2381629943847656
Accuracy : 33.65
Epoch: 3  loss:  2.22152042388916
Accuracy : 39.74
Epoch: 4  loss:  2.209625720977783
Accuracy : 43.44
Epoch: 5  loss:  2.201089859008789
Accuracy : 45.84
Epoch: 6  loss:  2.1950104236602783
Accuracy : 47.41
Epoch: 7  loss:  2.190706253051758
Accuracy : 48.33
Epoch: 8  loss:  2.1876566410064697
Accuracy : 48.87
Epoch: 9  loss:  2.185500383377075
Accuracy : 49.33
Epoch: 10  loss:  2.1839799880981445
Accuracy : 49.58
Epoch: 11  loss:  2.1829147338867188
Accuracy : 49.85
Epoch: 12  loss:  2.1821694374084473
Accuracy : 50.02
Epoch: 13  loss:  2.1816508769989014
Accuracy : 50.12
Epoch: 14  loss:  2.181293249130249
Accuracy : 50.16
```
Because the learning rate is decreasing, so the initial learning rate should be larger, or the loss of the model will not keep decreasing
# 2. Learning Rate
### lr = 1.0
1 is too large for this model, if we use 1 as our learning rate, the loss value will become `nan`, making it impossible to use gradient descend. 
[[MNIST#Device]]
以下的結果都有偏差，應該要跟其他參數的model進行比較，不能單純比較trainingloss跟testingloss的走向。如果單純看trainingloss跟testingloss的話：
因為dropout會讓model在訓練時當機，但是testing時不會，所以testing時的performance一定比較好，要將不同參
### lr = 0.001
![[2024-05-14_19-56.jpeg]]
### lr =0.001, gamma = 0.85
###### Result
```python 
Epoch: 1  loss:  1.8050339221954346
Accuracy : 70.27
Epoch: 2  loss:  1.0958017110824585
Accuracy : 79.03
Epoch: 3  loss:  0.7196364998817444
Accuracy : 83.53
Epoch: 4  loss:  0.5428182482719421
Accuracy : 85.54
Epoch: 5  loss:  0.44801145792007446
Accuracy : 86.63
Epoch: 6  loss:  0.39161646366119385
Accuracy : 87.24
Epoch: 7  loss:  0.3553548753261566
Accuracy : 87.81
Epoch: 8  loss:  0.3305912911891937
Accuracy : 88.31
Epoch: 9  loss:  0.3128986358642578
Accuracy : 88.59
Epoch: 10  loss:  0.29979708790779114
Accuracy : 88.74
Epoch: 11  loss:  0.28983640670776367
Accuracy : 88.89
Epoch: 12  loss:  0.2821024954319
Accuracy : 88.97
Epoch: 13  loss:  0.2759818136692047
Accuracy : 89.06
Epoch: 14  loss:  0.271062970161438
Accuracy : 89.15

```

### lr = 0.001, gamma =0.9
###### Result:
```python
Epoch: 1  loss:  1.8050339221954346
Testing Loss :  1.721859633922577
Accuracy : 70.27
Epoch: 2  loss:  1.0617951154708862
Testing Loss :  1.0171711921691895
Accuracy : 79.56
Epoch: 3  loss:  0.6717929840087891
Testing Loss :  0.7074876010417939
Accuracy : 84.11
Epoch: 4  loss:  0.4923369586467743
Testing Loss :  0.5724080592393875
Accuracy : 86.12
Epoch: 5  loss:  0.39813390374183655
Testing Loss :  0.5004087388515472
Accuracy : 87.22
Epoch: 6  loss:  0.3430500030517578
Testing Loss :  0.4565331727266312
Accuracy : 88.0
Epoch: 7  loss:  0.30794501304626465
Testing Loss :  0.427333565056324
Accuracy : 88.63
Epoch: 8  loss:  0.2840198576450348
Testing Loss :  0.4066690430045128
Accuracy : 88.88
Epoch: 9  loss:  0.2667250335216522
Testing Loss :  0.3913623049855232
Accuracy : 89.21
Epoch: 10  loss:  0.2536216378211975
Testing Loss :  0.37962366044521334
Accuracy : 89.52
Epoch: 11  loss:  0.2433442622423172
Testing Loss :  0.3703680634498596
Accuracy : 89.72
Epoch: 12  loss:  0.2350684404373169
Testing Loss :  0.3629102259874344
Accuracy : 89.87
Epoch: 13  loss:  0.22828719019889832
Testing Loss :  0.35679482370615007
Accuracy : 90.01
Epoch: 14  loss:  0.2226245105266571
Testing Loss :  0.3517079040408134
Accuracy : 90.12
```
The value of loss is still decreasing, I think higher initial learning rate should be better, since the learning rate will decay later on. 
###### loss
![[2024-05-14_22-22 1.jpeg]]

### lr = 0.01 gamma = 0.8 
###### Result:
set learning rate to 0.01, which is ten time higher then the original setting. 
```python 
Epoch: 1  loss:  0.718553775921464
Testing Loss :  0.33153082579374316
Accuracy : 90.21
Epoch: 2  loss:  0.30268856688444296
Testing Loss :  0.26780144944787027
Accuracy : 92.1
Epoch: 3  loss:  0.25741661074104655
Testing Loss :  0.23663205057382583
Accuracy : 93.19
Epoch: 4  loss:  0.23088516560055491
Testing Loss :  0.2163049191236496
Accuracy : 93.71
Epoch: 5  loss:  0.21288048521057565
Testing Loss :  0.20226366817951202
Accuracy : 94.15
Epoch: 6  loss:  0.19997134460890884
Testing Loss :  0.192160664498806
Accuracy : 94.49
Epoch: 7  loss:  0.19045616508419835
Testing Loss :  0.18481220155954362
Accuracy : 94.7
Epoch: 8  loss:  0.18332365914873444
Testing Loss :  0.1793310262262821
Accuracy : 94.87
Epoch: 9  loss:  0.17789715313132226
Testing Loss :  0.1752157501876354
Accuracy : 94.99
Epoch: 10  loss:  0.17373094702961603
Testing Loss :  0.172061238437891
Accuracy : 95.07
Epoch: 11  loss:  0.17049460710763042
Testing Loss :  0.16963952481746675
Accuracy : 95.16
Epoch: 12  loss:  0.16796742864787134
Testing Loss :  0.16775761917233467
Accuracy : 95.21
Epoch: 13  loss:  0.16598466700757109
Testing Loss :  0.16628799065947533
Accuracy : 95.23
Epoch: 14  loss:  0.16442165211307755
Testing Loss :  0.16512699723243712
Accuracy : 95.22
```
###### loss
![[Pasted image 20240514223126.png]]
### lr = 0.1 gamma = 0.8
###### Result
```python 
Epoch: 1  loss:  0.2674580568597634
Testing Loss :  0.13451431058347224
Accuracy : 95.66
Epoch: 2  loss:  0.09526688682321491
Testing Loss :  0.09110054075717926
Accuracy : 97.14
Epoch: 3  loss:  0.060734579786544104
Testing Loss :  0.07583533767610788
Accuracy : 97.51
Epoch: 4  loss:  0.042501832374921646
Testing Loss :  0.06861406052485108
Accuracy : 97.85
Epoch: 5  loss:  0.03165367339417633
Testing Loss :  0.0648601908236742
Accuracy : 97.93
Epoch: 6  loss:  0.024790470147873264
Testing Loss :  0.06202674778178334
Accuracy : 98.02
Epoch: 7  loss:  0.02024726385690286
Testing Loss :  0.06034850003197789
Accuracy : 98.12
Epoch: 8  loss:  0.017202350143991064
Testing Loss :  0.05905055571347475
Accuracy : 98.16
Epoch: 9  loss:  0.015110956977071516
Testing Loss :  0.05822489811107516
Accuracy : 98.16
Epoch: 10  loss:  0.013625567711382505
Testing Loss :  0.05749621903523803
Accuracy : 98.16
Epoch: 11  loss:  0.01254008717830978
Testing Loss :  0.05699918577447534
Accuracy : 98.16
Epoch: 12  loss:  0.011740550580570401
Testing Loss :  0.05665997266769409
Accuracy : 98.13
Epoch: 13  loss:  0.011138710901873107
Testing Loss :  0.05644219554960728
Accuracy : 98.13
Epoch: 14  loss:  0.010679798913770355
Testing Loss :  0.0563036996871233
Accuracy : 98.13

```
This model is facing [[Overfitting]] fitting issue
###### loss
![[2024-05-14_22-35.jpeg]]
Introducing weight decay to implement [[Regularization]]
### lr = 0.01 , gamma = 0.8 , weight_decay = 1e-3
###### Result
```python 
Epoch: 1  loss:  0.277956784405966
Testing Loss :  0.15278715267777443
Accuracy : 95.03
Epoch: 2  loss:  0.11431464328190316
Testing Loss :  0.11106984876096249
Accuracy : 96.63
Epoch: 3  loss:  0.0859337570999604
Testing Loss :  0.09379347376525402
Accuracy : 97.08
Epoch: 4  loss:  0.07215014514999825
Testing Loss :  0.08521966375410557
Accuracy : 97.4
Epoch: 5  loss:  0.06400781218086614
Testing Loss :  0.080057168379426
Accuracy : 97.58
Epoch: 6  loss:  0.05871515397676456
Testing Loss :  0.07671091854572296
Accuracy : 97.75
Epoch: 7  loss:  0.055083788141781
Testing Loss :  0.07439223565161228
Accuracy : 97.8
Epoch: 8  loss:  0.05249588279627454
Testing Loss :  0.07266897987574339
Accuracy : 97.91
Epoch: 9  loss:  0.050584485295411354
Testing Loss :  0.07140988521277905
Accuracy : 97.94
Epoch: 10  loss:  0.049124707209456685
Testing Loss :  0.0704663023352623
Accuracy : 97.95
Epoch: 11  loss:  0.04801111020319569
Testing Loss :  0.06966190300881862
Accuracy : 98.04
Epoch: 12  loss:  0.04713694950360169
Testing Loss :  0.06905243620276451
Accuracy : 98.08
Epoch: 13  loss:  0.04644905628764561
Testing Loss :  0.06857832279056311
Accuracy : 98.08
Epoch: 14  loss:  0.04590431654921123
Testing Loss :  0.06822349447757006
Accuracy : 98.08
```

###### loss 
![[Pasted image 20240514224250.png]]




### lr = 0.01 , gamma = 0.8 , weight_decay = 1e-3, dropout = 0.2
Dropout can randomly turn off some neural network.[[dropout|dropout]] 
###### Result
```python 
Epoch: 1  loss:  0.262087279510722
Testing Loss :  0.154629947245121
Accuracy : 95.14
Epoch: 2  loss:  0.11565225148401749
Testing Loss :  0.11251650974154473
Accuracy : 96.59
Epoch: 3  loss:  0.09027113858908256
Testing Loss :  0.10385909602046013
Accuracy : 96.79
Epoch: 4  loss:  0.07754731167312354
Testing Loss :  0.09026656225323677
Accuracy : 97.18
Epoch: 5  loss:  0.0695220501234592
Testing Loss :  0.0865807618945837
Accuracy : 97.3
Epoch: 6  loss:  0.06389038296151303
Testing Loss :  0.07843034062534571
Accuracy : 97.62
Epoch: 7  loss:  0.05850320028514464
Testing Loss :  0.0724745498970151
Accuracy : 97.72
Epoch: 8  loss:  0.0568054688018141
Testing Loss :  0.07024972140789032
Accuracy : 97.83
Epoch: 9  loss:  0.05405697682554693
Testing Loss :  0.06792319901287555
Accuracy : 97.92
Epoch: 10  loss:  0.052345232396418334
Testing Loss :  0.06785886380821467
Accuracy : 97.93
Epoch: 11  loss:  0.05082052705565225
Testing Loss :  0.065304596722126
Accuracy : 98.05
Epoch: 12  loss:  0.050160969608524904
Testing Loss :  0.0646391874179244
Accuracy : 98.07
Epoch: 13  loss:  0.04863378957339894
Testing Loss :  0.06471603848040104
Accuracy : 98.08
Epoch: 14  loss:  0.04799152143410782
Testing Loss :  0.06404446363449097
Accuracy : 98.07

```
###### Loss
![[Pasted image 20240514230855.png]]

### lr = 0.01 , gamma = 0.8 , weight_decay = 1e-3, dropout = 0.5
我一開始以為如果Dropout太大會砍掉一半左右的網路，感覺會讓模型無法判斷跟訓練，但是結果這樣可以防止模型overfitting，效果很好。
###### Result:
```python 
Epoch: 1  loss:  0.30628054587047365
Testing Loss :  0.17000885978341101
Accuracy : 94.6
Epoch: 2  loss:  0.1543461404937401
Testing Loss :  0.11468580551445484
Accuracy : 96.43
Epoch: 3  loss:  0.12765049452045515
Testing Loss :  0.10020112283527852
Accuracy : 97.03
Epoch: 4  loss:  0.11088329233052427
Testing Loss :  0.0938534114509821
Accuracy : 97.04
Epoch: 5  loss:  0.09866405091832267
Testing Loss :  0.09082319475710392
Accuracy : 97.19
Epoch: 6  loss:  0.09186284423225931
Testing Loss :  0.07965642940253019
Accuracy : 97.65
Epoch: 7  loss:  0.08479519482694074
Testing Loss :  0.07804429065436125
Accuracy : 97.59
Epoch: 8  loss:  0.08054664874745252
Testing Loss :  0.07350067999213934
Accuracy : 97.76
Epoch: 9  loss:  0.0760940246225428
Testing Loss :  0.07128553818911314
Accuracy : 97.81
Epoch: 10  loss:  0.07379809713441887
Testing Loss :  0.07071884609758854
Accuracy : 97.88
Epoch: 11  loss:  0.07239822799296204
Testing Loss :  0.06873963177204132
Accuracy : 97.94
Epoch: 12  loss:  0.0700221089889476
Testing Loss :  0.06781211365014314
Accuracy : 97.98
Epoch: 13  loss:  0.06873371179001148
Testing Loss :  0.06680391989648342
Accuracy : 98.03
Epoch: 14  loss:  0.06733019679197982
Testing Loss :  0.06702467203140258
Accuracy : 98.05
```

###### Loss 
![[Pasted image 20240514231357.png]]
### dropout = 0.8 
![[Pasted image 20240514233225.png]]
The loss is not decreasing very well. 


# 2. Using Adam Optimizer
- learning rate: 0.001
- weight decay : not set yet 
- dropout rate 0.8
![[Pasted image 20240514234134.png]]

> [!NOTE] Learning Rate
> Larger learning rate will lead to poor performance using Adam optimizer. 
> 
###### Result: 
```python 
Epoch: 1  loss:  59.650719285392555
Testing Loss :  5.549348711967468
Accuracy : 9.82
Epoch: 2  loss:  10.298129650321343
Testing Loss :  4.505307960510254
Accuracy : 10.31
Epoch: 3  loss:  3.4744214927718073
Testing Loss :  4.242453622817993
Accuracy : 10.31
Epoch: 4  loss:  2.3177377829419523
Testing Loss :  4.497348642349243
Accuracy : 10.32
Epoch: 5  loss:  2.30967761306112
Testing Loss :  4.496854138374329
Accuracy : 10.3
Epoch: 6  loss:  2.3089389117287675
Testing Loss :  4.496446537971496
Accuracy : 10.3
Epoch: 7  loss:  2.30823751070352
Testing Loss :  4.4961800336837Required libruaries of ParaSeis765
Accuracy : 10.3
Epoch: 8  loss:  2.307582588592318
Testing Loss :  4.496067380905151
Accuracy : 10.3
Epoch: 9  loss:  2.306976157973316
Testing Loss :  4.496035265922546
Accuracy : 9.57
Epoch: 10  loss:  2.306417241025327
Testing Loss :  4.495971822738648
Accuracy : 9.57
Epoch: 11  loss:  2.3059040145325
Testing Loss :  4.4958092212677006
Accuracy : 9.57
Epoch: 12  loss:  2.305434732548972
Testing Loss :  4.495547986030578
Accuracy : 9.57
Epoch: 13  loss:  2.305007434348816
Testing Loss :  4.495229721069336
Accuracy : 9.57
Epoch: 14  loss:  2.304619714395324
Testing Loss :  4.494902563095093
Accuracy : 9.57
>```




### Dropout set to 0.2
![[Pasted image 20240515000017.png]]
The test loss is not decreasing, so I set drop out to larger value to avoid overfitting. 
### Dropout set to 0.5
![[Pasted image 20240515000123.png]]

### Dropout set to 0.8 
![[Pasted image 20240515000358.png]]

# 3. Convolution Model 
Using Convolution Model greatly enhance the accuracy of digit classification. The accuracy is up to 99.41% at the end. 
###### Result: 
```python 
Epoch: 1  loss:  0.38516100652779994
Testing Loss :  0.07417778559029102
Accuracy : 97.85
Epoch: 2  loss:  0.13564339445244625
Testing Loss :  0.04538081977516413
Accuracy : 98.48
Epoch: 3  loss:  0.10288692498256935
Testing Loss :  0.038665673788636924
Accuracy : 98.83
Epoch: 4  loss:  0.08660195264110176
Testing Loss :  0.03503405964002013
Accuracy : 98.83
Epoch: 5  loss:  0.07723655981328914
Testing Loss :  0.03152931081131101
Accuracy : 98.87
Epoch: 6  loss:  0.07231190276872326
Testing Loss :  0.027586940582841633
Accuracy : 99.05
Epoch: 7  loss:  0.06350235230080149
Testing Loss :  0.027135903108865024
Accuracy : 99.04
Epoch: 8  loss:  0.0618595177578161
Testing Loss :  0.023327867640182375
Accuracy : 99.25
Epoch: 9  loss:  0.053117936011465854
Testing Loss :  0.02377397094387561
Accuracy : 99.15
Epoch: 10  loss:  0.054007979424564596
Testing Loss :  0.022024831245653333
Accuracy : 99.16
Epoch: 11  loss:  0.04807637642146208
Testing Loss :  0.021923902677372098
Accuracy : 99.21
Epoch: 12  loss:  0.046694290571360056
Testing Loss :  0.021878561237826943
Accuracy : 99.29
Epoch: 13  loss:  0.04437653749862794
Testing Loss :  0.022098707384429872
Accuracy : 99.3
Epoch: 14  loss:  0.04428922246181856
Testing Loss :  0.02106117832008749
Accuracy : 99.24
Epoch: 15  loss:  0.04410278461436353
Testing Loss :  0.020311500248499214
Accuracy : 99.26
Epoch: 16  loss:  0.038606060409637565
Testing Loss :  0.020667731971479954
Accuracy : 99.25
Epoch: 17  loss:  0.03850508906634593
Testing Loss :  0.01908551671076566
Accuracy : 99.33
Epoch: 18  loss:  0.036747972074212744
Testing Loss :  0.01924531136173755
Accuracy : 99.39
Epoch: 19  loss:  0.03670361579000478
Testing Loss :  0.01878916665446013
Accuracy : 99.41
Epoch: 20  loss:  0.03627882353118884
Testing Loss :  0.018121643806807697
```

###### loss: 
![[Pasted image 20240515082725.png]]


