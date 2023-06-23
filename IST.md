所有灰色棋子存储在 greyPieces的列表中，self.greyPieces.append((idTag, newChecker))
idTag是棋子标号，newChecker是棋子object，可以通过newChecker.getRow 等获取棋子信息

1. **Introduction**
    
    - Brief overview of the project and its objectives.
2. **Gameplay**
    
    - Description of the interactive checkers gameplay.
    - Explanation of the different levels of AI cleverness and how the user can adjust them.
3. **Search Algorithm**
    
    - Explanation of the state representation used in the game.
    - Description of the successor function and how it generates AI moves.
    - Explanation of the Minimax evaluation and Alpha-Beta pruning.
    - Discussion on the use of heuristics in the game.
4. **Validation of Moves**
    
    - Description of how the AI ensures no invalid moves.
    - Explanation of how user moves are checked for validity.
    - Discussion on how the game handles invalid user moves and provides explanations.
    - Explanation of the forced capture rule.
5. **Other Features**
    
    - Description of the multi-step capturing moves for both the user and the AI.
    - Explanation of the king conversion at baseline.
    - Discussion on the regicide rule.
    - Description of the help facility that provides hints about available moves.
6. **GUI-Specific Features**
    
    - Description of the graphical board representation.
    - Explanation of how the interface updates the display after completed moves.
    - Discussion on the helpful GUI instructions provided.
    - Description of the mouse interaction features.
    - Explanation of how the GUI pauses to show the intermediate legs of multi-step moves.
    - Description of the dedicated display of the rules.
7. **Challenges and Solutions**
    
    - Detailed analysis of the five most challenging problems encountered during the development of the program and how they were overcome.
8. **Conclusion**
    
    - Summary of the project and its outcomes.
    - Reflections on the development process and final product.
9. **References**
    
    - Cite any resources or references used during the development of the project.


在我写的跳棋游戏中，我为玩家设置了三种游戏难度：简单、普通、困难。游戏的难度级别是通过操作minimax算法中搜索树的深度以及alpha和beta值来调整的。搜索树的深度本质上决定了AI在做出决策之前会考虑多少步，而alpha和beta值用于修剪搜索树以优化AI的决策过程。
简单关卡:在简单关卡中，AI的思考深度只有1层(max_depth = 1)， alpha和beta值分别设置为无穷大和负无穷大，这样设置alpha-beta值的目的是让AI舍弃得分最高的步骤，选择得分最低的步骤，因此难度最低
普通关卡:在普通关卡，AI的思考深度同样为1层(max_depth = 1)，但alpha和beta值分别被设置为负无穷大和正无穷大。相比简单关卡，AI的决策过程得到优化，会选择得分最高的步骤，因此难度相较于简单提高了不少。
困难关卡：在硬关卡中，AI的思考深度是3层(max_depth = 3)， alpha和beta值分别设置为负无穷大和正无穷大。这意味着AI在决定移动之前会考虑更广泛的未来可能情况，即使对经验丰富的玩家来说，它也是一个难以对付的对手。
用户随时可以通过游戏的菜单栏中的Difficulty选项进行三种难度的调整。这让所有技能水平的玩家都能享受游戏，并随着他们的进步逐渐增加挑战。