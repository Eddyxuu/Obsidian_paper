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


在代码中我通过start_new_game()函数创建了一个名为pieces_state的2D列表来进行适当的状态表示，这种表示有效地捕获了游戏板的当前状态，允许在搜索过程中轻松操作和跟踪游戏状态。
pieces_state列表被初始化和更新，以反映棋子在棋盘上的位置和类型。2D列表中的每个元素代表棋盘上的一个单元格，每个单元格中的值表示该位置的状态。下面是对pieces_state中的值的解释:
0:空单元格
1:人类玩家的常规棋子
2:人类玩家的国王棋子(提升普通棋子)
3: AI玩家的常规棋子
4: AI玩家的王棋子(提升的常规棋子)
在代码的scan()函数中，会通过扫描pieces_state来查看当前的棋局状态，支持在游戏和搜索过程中进行有效的操作和评估

程序中后续函数通过评估所有可能的移动，根据评估分数选择最佳移动，并在游戏期盼上执行该移动来生成AI移动。AI的移动是通过computer_turn()函数生成的。此功能控制AI玩家的回合，并包含基于对可能的移动评估为AI选择最佳移动的逻辑。computer_turn()函数首先调用evaluate_ai_all_possible()函数来评估AI玩家的所有可能的移动。当评估完成时，该函数根据评估分数选择最佳移动。如果存在多个具有相同最高分的移动，则会从这些选项中随机选择一个移动。然后使用move()函数执行选定的移动。get_ai_all_possible()函数负责为AI玩家检索所有可能的移动。它首先通过调用check_move_ai_jump_allPiece()来尝试捕获对手的棋子，check_move_ai_jump_allPiece()检查捕获的移动。如果没有可用的捕获移动，它调用check_move_ai_normalMove_allPiece()来获取所有正常的非捕获移动。

启发式还能提升
While the current implementation relies on a basic heuristic, more advanced heuristics could be incorporated to consider factors such as piece mobility, king positions, and board control. These additional heuristics can enhance the AI's decision-making process and lead to more strategic gameplay.
当前启发式的能力不足
高级的启发式可以被纳入考虑诸如棋子移动、国王位置和棋盘控制等因素。但是我在编写启发式的逻辑的过程中出现了问题，导致当前只有最基础的启发式能力


代码限定了游戏规则，因此不会玩家不会出现违反规则的无效移动。只有一种可能即玩家没有注意到棋盘上有可以跳吃的棋子而想选择其他棋子的情况，代码专门为这种情况设计了人性化弹窗提示如Figure 2所示。


Regicide: if a normal piece manages to capture a king, it is instantly crowned king
The code implements Regicide in the move() function when determining that the jumped piece is the opponent's king. Change the properties of the current piece by pieces_state to make it king


The code does include a built-in help facility that provides hints about available moves.
The code calls the update_highlight_avaliableTile() function via the position_2() function to highlight the possible progressions of the selected piece, as shown in Figure 3


代码使用了Tkinter模块进行GUI展示，其中使用了canvas.create_oval生成棋子，canvas.create_rectangle生成期盼


代码通过move()函数来控制棋子移动，在move()中有draw()进行GUI同步更新，使棋子在完成每一步操作之后都可以同步更新在GUI上。

