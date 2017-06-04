
import numpy as np
import itertools
import scipy.misc
import matplotlib.pyplot as plt


class GameItem():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class GameWorld():
    def __init__(self, size, partial):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        #
        self.reset()        
        #
        plt.figure()
        plt.imshow(self.mapToPic(self.state), interpolation="nearest")
        plt.show()
        #
        
    def reset(self):
        self.objects = []
        hero = GameItem(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        food = GameItem(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(food)
        pois = GameItem(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(pois)
        food2 = GameItem(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(food2)
        pois2 = GameItem(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(pois2)
        food3 = GameItem(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(food3)
        food4 = GameItem(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(food4)
        #    
        return self.renderWorld()
        #
    
    def renderWorld(self):
        # further simplification
        # a = np.ones([self.sizeY, self.sizeX, 3])
        a = np.ones([self.sizeY+2, self.sizeX+2, 3]);
        a[1:-1,1:-1,:] = 0;
        hero = None;
        for item in self.objects:
            a[item.y+1:item.y+item.size+1,\
              item.x+1:item.x+item.size+1,\
              item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y+3,hero.x:hero.x+3,:]                    
        #
        self.state = a;
        return a;
        #
    
    def mapToPic(self, state):
        b = scipy.misc.imresize(state[:,:,0],[84,84,1],interp='nearest')
        c = scipy.misc.imresize(state[:,:,1],[84,84,1],interp='nearest')
        d = scipy.misc.imresize(state[:,:,2],[84,84,1],interp='nearest')
        pic = np.stack([b,c,d], axis=2)
        #        
        return pic;
    
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]
        
    def step(self, action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderWorld()
        #
        return state,(reward+penalty),done
        
    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.0
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1     
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize
    
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        #ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(GameItem(self.newPosition(),1,1,1,1,'goal'))
                else: 
                    self.objects.append(GameItem(self.newPosition(),1,1,0,-1,'fire'))
                return other.reward,False
        else:
            return 0.0, False
        #if ended == False:
        #    return 0.0,False

    