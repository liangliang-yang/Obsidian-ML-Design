
https://leetcode.com/problems/find-peak-element/solutions/788474/general-binary-search-thought-process-4-templates/

### Find the _First True

1. **Find the _First True_**:
    
    **T** T T T T F F F or F F F F **T** T T T  
    We always try to go the left when finding the First True and the template will be like :
    
    ```sql
    while left < right :
    	if condition is true:
    		right = mid 
    	else:
    		left = mid +1
    ```
    
    **Thought Process for this template :** For FFFFTTTT, if mid is second last True , then it is a possible ans but we need to keep searching left because we want First True or minimal True for it . We move to left part by doing right = mid where mid is a potential solution and throw away the right part
    
```python
def findFirstTrue(self, nums):
	left= 0 
	right = len(nums)-1
	while left < right:
		mid = left + (right-left)//2 # left-biased midpoint, standard way

		if nums[mid] is True: 
			right = mid # first True must inside [left, mid]
		else:
			left = mid+1 # first True must inside [mid+1, right]

	return left
```

### Find the _Last True

1. **Find the _Last True_**:
    
    T T T T **T** F F F or F F F F T T T T **T**
    
    We always try to go the right when finding the Last True and the template will be like :
    
    ```sql
    while left < right :
    	if condition is true:
    		left = mid 
    	else:
    		right = mid - 1
    ```
    
    **Thought Process for this template :** For T T T T T F F F, if mid is First True , then it is a possible ans but we need to keep searching right because we want Last True or Maximal True for it . We move to right half by doing left = mid where mid is a potential solution and throw away the left part

```python
def findLastTrue(self, nums):
	left= 0 
	right = len(nums)-1
	while left < right:
		mid = right - (right-left)//2 # right-biased midpoint
		# if mid is calculated as left+(right-left)//2, there could be
		# cases where left is not updated, leading to an infinite loop

		if nums[mid] is True: 
			left = mid # Last True must inside [mid, right]
		else:
			right = mid-1 # Last True must inside [left, mid-1]

	return left
```

