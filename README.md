# Monte-Carlo-Algorithm

## Task
1. Implement MC-Epsilon Greedy and compare it with MC-Exploring Start.
2. In this regard, from s13 a drone can take our robot to s1, 
and from s9 same drone can take it to s6. (note that you need to add a new action called fly).

## Code Explain
### Implement MC-Epsilon Greedy - Monte_Carlo_Algorithm.py
The main difference between class “MC_Exploring_Start” and class “MC_Epsilon_Greedy” 
in my code is function “policy” part. 

After we get the “valid_actions” list for the current state coordinate, 
I added an if...else... statement and parameter “EPSILON” to implement Epsilon Greedy concept. 
Parameter “EPSILON” defined as 0.1.
By using random() function to generate a float number between 0 and 1, 
for example let’s call it “rand_num”. 
If rand_num is bigger than the EPSILON, 
then choose the action which had maximum value in this state. 
If rand_num is smaller than the EPSILON, randomly choose an action from valid_actions.
And the final optimal policy also shown as below.  

![optimal policy](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8bf02770-8927-4b29-981a-1c2aed1e2c91/optimal_policy.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221004%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221004T160901Z&X-Amz-Expires=86400&X-Amz-Signature=4e54e27cfe12e73e37867668bc4b15910c54fc739d156356bdf0dc207e993855&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22optimal_policy.png%22&x-id=GetObject)

By doing so, we can see the converged result from the figure shown below. 
Compare “MC Epsilon Greedy” with “MC Exploring Start”, 
MC Epsilon Greedy has 10% chance to randomly select an action, 
so its line on chart is less compact to MC Exploring Start. 
If we decrease the EPSILON to 0.01, which means 1% chance to randomly select an action, 
then we will get closer range between MC Exploring Start and MC Epsilon Greedy.  

![epsilon 0.1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/6450ef6d-b24a-441a-83d4-4fec1080b24e/epsilon_01.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221004%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221004T161016Z&X-Amz-Expires=86400&X-Amz-Signature=47204babbfe331a3f0c6ee23824c7f25f200a1832113883a02f6bd4e53b1881e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22epsilon_01.png%22&x-id=GetObject)
![epsilon 0.01](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/120790c7-730a-46c2-b066-a43145e7fe2a/epsilon_001.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221004%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221004T161019Z&X-Amz-Expires=86400&X-Amz-Signature=338adb321b20dd8ace23bbc49652962187b1b57f062758067ac02732a4852bc8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22epsilon_001.png%22&x-id=GetObject)

### Add drone - MC_with_drone.py
First, added “fly” action into action dictionary, 
with [0, 0] value (later will give it different value based on different state, s9 or s13).  

Second, at function “transfer_state”, I determined whether the input state_coordinate is 
s9[1, 3] or s13[2, 4] or other state coordinate.   

If state_coordinate is s9[1, 3], then see what’s the random input action are given. 
If the random action is “fly”, then the drone will take the robot from s9[1, 3] to s1[1, 0], 
which means robot need to stay on the same row but go left for 3 columns. 
As the code shown below, <state_coordinates + [0, -3]>. Otherwise, 
if the random action is not fly but other actions (up, down, right, left), 
then just add the moving action value according to the direction dictionary.   

Same concept for s13, if the input state_coordinate is s13[2, 4], 
and the random action is “fly”. Then, the robot will move from s13[2, 4] to s1[0, 1], 
which means the robot need to go up for 1 row and go left for 3 columns. 
As the code shown below, <state_coordinates + [-2, -3]>. 
Otherwise, if the random action is not fly but other action (up, down, right, left), 
then just add the moving action value according to the direction dictionary.   

And if the input state_coordinate is neither s9 nor s13, 
then it must be other state in range s1, s2, ..., s8, s10, s11, s12, s14, ..., s20. 
In this case, just add the moving action value according to the direction dictionary. 
By doing so, we can calculate the sum of state_coordinate and action value, 
and get the next state coordinate.  

After getting the coordinate for the next state, 
we determine whether next state coordinate is out of the board or not. 
Range from 0 to 3 for row, and 0 to 5 for column. 
If next state coordinate is out the board, 
then just return the value of the input state coordinate as we assign as “current_state_coordinate” at the very beginning.  

At last, we can assure that the next state coordinate is in the grid, 
then we need to check whether the next state is a “wall” or not. 
If next state is wall, apparently we can not choose it as our next step, 
therefore just return the “current_state_coordinate”.   

The optimal policy is shown below, at state s9[1, 3] and s13[2, 4] both will take the fly action.
Even s10[1, 5] and s18[3, 3] also influence by s13, 
both of the actions are try to go to state s13 for shorter route.   

![optimal_policy_with_drone](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/a4f12674-a698-4cbe-8c8e-8f2fdbde301c/optimal_policy_with_drone.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221004%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221004T162113Z&X-Amz-Expires=86400&X-Amz-Signature=c4a8663fdd20ed91279497081b5dc99426cda1d250a61afd357dcb4aed7112a5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22optimal_policy_with_drone.png%22&x-id=GetObject)

