
#(Almost same with Matris.py , following is the main difference)

       initial  = True
        while True:
            try:
                timepassed = clock.tick(100) #50
                if self.matris.update((timepassed / 1000.) if not self.matris.paused else 0):
                    self.redraw()
                    if(self.matris.new_tetromino_here):
                        #reward = self.matris.score - current_score - 320
                        bot.set_matrix(self.matris.get_matrix(),False)
                        bot.setScore(self.matris.score,False)
                        if not initial:
                            reward = bot.calcReward(table)
                            bot.update_Q_matrix(table,self.matris.current_tetromino,reward)
                        shape,position,table = bot.best_move(self.matris.current_tetromino)
                        bot.set_matrix(self.matris.get_matrix(),True)
                        bot.setScore(self.matris.score,True)
                        moves = bot.move_tetr(self.matris.current_tetromino,shape,position,table)
                        #print(self.matris.current_tetromino.name)
                        self.matris.set_events(moves)
                        self.matris.new_tetromino_here = False
                        initial = False
           except GameOver:
