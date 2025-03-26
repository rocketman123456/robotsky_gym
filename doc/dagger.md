Dagger is actually pretty simple. The motivation for it is that if you do straight supervised learning on data from an expert the error from not perfectly fitting the training data will compound over time and your policy will quickly get further and further away from the expert trajectory, until eventually it's somewhere it's never seen in the training data and will behave unpredictably.

Dagger fixes this compounding error issue by having a step in the training regime where the policy runs for a while, and then the expert "corrects" the policies mistakes and these corrections get added to the training data. The result is that the policy is taught how to correct it's own bad behavior, and stay close to the expert's trajectory.

It's basically a loop like this:

1. Gather some (state, action) data from an expert and train a policy on it.

2. Run the resulting policy

3. Have the expert label what it would have done at all the states the policy visited. Add this data to the training data.

4. Train the policy on the new, aggregated, data.

5. Goto 2
