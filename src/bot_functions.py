

import numpy as np

dict_tetr_len = {"long":[4,1,4,1],
                 "square" : [2,2,2,2],
                 "hat" : [3,2,3,2],
                 "left_snake" : [3,2,3,2],
                 "right_snake" : [3,2,3,2],
                 "left_gun" : [3,2,3,2],
                 "right_gun" : [3,2,3,2]}

dict_tetr_shape = {"long":2,
                 "square" : 1,
                 "hat" : 4,
                 "left_snake" : 2,
                 "right_snake" : 2,
                 "left_gun" : 4,
                 "right_gun" : 4}

def drop_tetr(matrix,tetromino,col,rotation):
    m = matrix.copy()
    game_over = False
    if tetromino.name == 'long':
        if rotation % 2 == 0  :
            if np.sum(m[:,col:col+4]) == 0:
                m[-1:,col:col+4] = 1
            else:
                q = m[:,col] + m[:,col+1] + m[:,col+2] + m[:,col+3]
                q = (q>=1)
                i = q.argmax(axis = 0)
                if i - 2 < 1:
                    game_over = True
                else:
                    m[i-1:i,col:col+4] = 1
        else:
            if np.sum(m[:,col]) == 0:
                m[-4:,col] = 1
            else:
                q = m[:,col]
                i = q.argmax(axis = 0)
                if i - 2 < 4:
                    game_over = True
                else:
                    m[i-4:i,col] = 1
            
            
    elif tetromino.name == 'square':
        if np.sum(m[:,col:col+2]) == 0:
            m[-2:,col:col+2] = 1
        else:
            q = m[:,col] + m[:,col+1]
            q = (q>=1)
            i = q.argmax(axis = 0)
            if i - 2 < 2:
                game_over = True
            else:
                m[i-2:i,col:col+2] = 1
            
              
    elif tetromino.name == 'hat':
          if rotation % 4 == 0:
                if np.sum(m[:,col:col+3]) == 0:
                    m[-1,col:col+3] = 1
                    m[-2,col+1]  = 1
                else:
                    q = np.sum(m[:,col:col+3],axis = 1)
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i-2,col+1]  = 1       
          elif rotation % 4 == 1:
              if np.sum(m[1:,col] + m[:-1,col+1]) == 0:
                    m[-1,col] = 1
                    m[-2,col:col+2]  = 1
                    m[-3,col]  = 1
              else:
                    q = m[1:,col] + m[:-1,col+1]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 2:
                        game_over = True
                    else: 
                        m[i-2:i+1,col] = 1
                        m[i-1,col+1] = 1
          elif rotation % 4 == 2:
                if np.sum(m[:-1,col] + m[1:,col+1] + m[:-1,col+2]) == 0:
                    m[-2,col:col+3] = 1
                    m[-1,col+1]  = 1
                else:
                    q = m[:-1,col] + m[1:,col+1] + m[:-1,col+2]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i,col+1] = 1
          elif rotation % 4 == 3:
              if np.sum(m[1:,col+1] + m[:-1,col]) == 0:
                    m[-1,col+1] = 1
                    m[-2,col:col+2]  = 1
                    m[-3,col+1]  = 1
              else:
                    q = m[1:,col+1] + m[:-1,col]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 2:
                        game_over = True
                    else: 
                        m[i-2:i+1,col+1] = 1
                        m[i-1,col] = 1   
                      
    elif tetromino.name == 'right_snake':
            if rotation % 2 == 0:
                  if np.sum(m[1:,col] + m[1:,col+1] + m[:-1,col+2]) == 0:
                        m[-1,col:col+2] = 1
                        m[-2,col+1:col+3]  = 1
                  else:
                        q = m[1:,col] + m[1:,col+1] + m[:-1,col+2]
                        q = (q>=1)
                        i = q.argmax(axis = 0)
                        if i - 2 < 1:
                            game_over = True
                        else: 
                            m[i-1,col+1:col+3] = 1
                            m[i,col:col+2] = 1   
            elif rotation % 2 == 1: 
                  if np.sum(m[1:,col+1] + m[:-1,col]) == 0:
                        m[-1,col+1] = 1
                        m[-2,col:col+2]  = 1
                        m[-3,col] = 1
                  else:
                        q = m[1:,col+1]  + m[:-1,col]
                        q = (q>=1)
                        i = q.argmax(axis = 0)
                        if i - 2 < 2:
                            game_over = True
                        else: 
                            m[i-2:i,col] = 1
                            m[i-1:i+1,col+1] = 1  
            
    elif tetromino.name == 'left_snake':
            if rotation % 2 == 0:
              if np.sum(m[1:,col+1] + m[1:,col+2] + m[:-1,col]) == 0:
                    m[-2,col:col+2] = 1
                    m[-1,col+1:col+3]  = 1
              else:
                    q = m[1:,col+1] + m[1:,col+2] + m[:-1,col]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i,col+1:col+3] = 1
                        m[i-1,col:col+2] = 1   
            elif rotation % 2 == 1: 
              if np.sum(m[1:,col] + m[:-1,col+1]) == 0:
                    m[-1,col] = 1
                    m[-2,col:col+2]  = 1
                    m[-3,col+1] = 1
              else:
                    q = m[1:,col]  + m[:-1,col+1]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 2:
                        game_over = True
                    else: 
                        m[i-2:i,col+1] = 1
                        m[i-1:i+1,col] = 1    

                      
    elif tetromino.name == 'left_gun':
         if rotation%4 == 0:
              if np.sum(m[:,col:col+3]) == 0:
                    m[-1,col:col+3] = 1
                    m[-2,col]  = 1
              else:
                    q = m[:,col] + m[:,col+1] + m[:,col+2]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 2:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i-2,col] = 1               
         if rotation%4 == 1:
              if np.sum(m[2:,col] + m[:-2,col+1]) == 0:
                    m[-1,col] = 1
                    m[-2,col]  = 1
                    m[-3,col:col+2] = 1
              else:
                    q = m[2:,col] + m[:-2,col+1]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1:i+2,col] = 1
                        m[i-1,col+1] = 1                     
         if rotation%4 == 2:
              if np.sum(m[:-1,col] + m[:-1,col+1] + m[1:,col+2]) == 0:
                    m[-1,col+2] = 1
                    m[-2,col:col+3]  = 1
              else:
                    q = m[:-1,col] + m[:-1,col+1] + m[1:,col+2]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i,col+2] = 1               
         if rotation%4 == 3:
              if np.sum(m[:,col:col+2]) == 0:
                    m[-3,col+1] = 1
                    m[-2,col+1]  = 1
                    m[-1,col:col+2] = 1
              else:
                    q = m[:,col] + m[:,col+1]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 3:
                        game_over = True
                    else: 
                        m[i-3:i,col+1] = 1
                        m[i-1,col] = 1             
              
    elif tetromino.name == 'right_gun':
         if rotation%4 == 0:
              if np.sum(m[:,col:col+3]) == 0:
                    m[-1,col:col+3] = 1
                    m[-2,col+2]  = 1
              else:
                    q = m[:,col] + m[:,col+1] + m[:,col+2]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 2:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i-2,col+2] = 1               
         if rotation%4 == 3:
              if np.sum(m[2:,col+1] + m[:-2,col]) == 0:
                    m[-1,col+1] = 1
                    m[-2,col+1]  = 1
                    m[-3,col:col+2] = 1
              else:
                    q = m[2:,col+1] + m[:-2,col]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1:i+2,col+1] = 1
                        m[i-1,col] = 1                     
         if rotation%4 == 2:
              if np.sum(m[:-1,col+1] + m[:-1,col+2] + m[1:,col]) == 0:
                    m[-1,col] = 1
                    m[-2,col:col+3]  = 1
              else:
                    q = m[:-1,col+2] + m[:-1,col+1] + m[1:,col]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 1:
                        game_over = True
                    else: 
                        m[i-1,col:col+3] = 1
                        m[i,col] = 1               
         if rotation%4 == 1:
              if np.sum(m[:,col:col+2]) == 0:
                    m[-3,col] = 1
                    m[-2,col]  = 1
                    m[-1,col:col+2] = 1
              else:
                    q = m[:,col] + m[:,col+1]
                    q = (q>=1)
                    i = q.argmax(axis = 0)
                    if i - 2 < 3:
                        game_over = True
                    else: 
                        m[i-3:i,col] = 1
                        m[i-1,col+1] = 1   
    return m,game_over


def sim_board(matrix):
    lines_cleared = 0
    for i in range(matrix.shape[0]):
        if sum(matrix[i,:]) == 10:
            lines_cleared = lines_cleared + 1
            matrix = np.delete(matrix,i,0)
            new_line = np.zeros((1,matrix.shape[1]))
            matrix = np.append(new_line,matrix,0)
    return (lines_cleared,matrix)
