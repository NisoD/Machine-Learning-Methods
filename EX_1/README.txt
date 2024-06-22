README
ex_1 {207447970}
Daniel Nisnevich
Name: Daniel Nisnevich
	
Scenario 1 Results
Number of times best prophet selected:  74
Average test error of selected prophet:  0.76
Average approximation error: 0.2
Average estimation error:  0.05200000000000002

Scenario 2 Results:
Number of times best prophet selected:  70
Average test error of selected prophet:  0.79
Average approximation error: 0.2
Average estimation error:  0.060000000000000026
Even though we 10x the amount of games we se no significant changes, the fake prophet won 4 more games than before slight changes in test error and average estimation error.

Scenario 3 Results:
Number of times best prophet selected:  24
Average test error of selected prophet:  0.28
Average approximation error:  0.5195802006372554
Average estimation error:  0.7210388097122995
We have a lot more prophets, we see a drastic decrease in times of choosing the true prophet.
Also, average approximating error has significantly increased together with average estimation error.
About changing the distribution to [0,0.5]  the estimation error and approximation error would both decrease because prophets would tend to be more correct and the difference would be smaller.





Scenario 4 Results:
Number of times best prophet selected:  0
Average test error of selected prophet:  0.29
Average approximation error:  0.48157753142823606
Average estimation error:  0.7108549816250715
Number of times almost best prophet selected:  100

Okay I have a problem I guess it is with the way I check which prophet won, and the errors matrix.
I deduce that the answer should be 100 and the estimation error and approximation error should be significantly smaller.

Scenario 5 Results:
[[(0.79, 0.16236135844202015) (0.87, 0.16358876937842182)
  (0.85, 0.1630432534066877) (0.83, 0.16413428535015584)]
 [(0.8, 0.10451723110587077) (0.9, 0.10869405268425952)
  (0.88, 0.10797508399486329) (0.89, 0.10797927493824501)]
 [(0.9, 0.12315354363042383) (0.83, 0.13640217855641193)
  (0.85, 0.121064773151695) (0.93, 0.12038203926089273)]
 [(0.86, 0.1385054218226246) (0.9, 0.12434505201726356)
  (0.83, 0.14231315552409196) (0.89, 0.13141827929811906)]]


Scenario 6 Results:
Average approximation error for class 1:  0.46097848046243195
Average approximation error for class 2:  0.5489781789435793
Average estimation error for class 1:  0.57
Average estimation error for class 2:  0.53
