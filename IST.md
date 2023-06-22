所有灰色棋子存储在 greyPieces的列表中，self.greyPieces.append((idTag, newChecker))
idTag是棋子标号，newChecker是棋子object，可以通过newChecker.getRow 等获取棋子信息

get_ai_possible_moves
	遍历greyPieces
	for checker in self.greyPieces:
		checker_object = checker[1]