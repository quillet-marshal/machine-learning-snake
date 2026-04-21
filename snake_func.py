def play(snakeDict=None, apple=None, action=None, tilesWide=20, n_observations=20):
    import random

    class shape:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

        def get_position(self):
            return (self.x, self.y)

        def set_position(self, coordinates):
            self.x = coordinates[0]
            self.y = coordinates[1]

        position = property(get_position, set_position)

    def moveSnake(movement): # moves whole snake one space along its path
        for i in range(len(snakeDict) - 1, 0, -1):
            snakeDict[i].position = snakeDict[i - 1].position

        match movement:
            case "up":
                snakeDict[0].y += tSize
            case "left":
                snakeDict[0].x -= tSize
            case "down":
                snakeDict[0].y -= tSize
            case "right":
                snakeDict[0].x += tSize

    def randomCord(): # returns a random valid coordinate
        upperLimit = int(windowSize - tSize)
        return random.randrange(0, upperLimit + 1, tSize)

    def spawnApple(): # spawns apple in an empty space
        validSpawn = False
        while not validSpawn:
            apple.x = randomCord()
            apple.y = randomCord()
            for i in range(0, len(snakeDict)):
                if apple.position == snakeDict[i].position:
                    # print("DEBUG: Apple tried to spawn on top of snake segment ", i)
                    validSpawn = False
                    break
                else:
                    validSpawn = True

    def handleInput(): # allows ML program to play the game
        global learningValue
        learningValue = -1
        global gameOver
        gameOver = False

        global offScreen
        offScreen = False
        global selfCollision
        selfCollision = False

        if action != None:
            if action == 0: moveSnake("up")
            elif action == 1: moveSnake("left")
            elif action == 2: moveSnake("down")
            elif action == 3: moveSnake("right")

            if snakeDict[0].position == apple.position:
                spawnApple()
                snakeDict[len(snakeDict)] = shape(0 - tSize, 0 - tSize, tSize, tSize)
                learningValue = int(20 * tilesWide) #100 + ((tilesWide * len(snakeDict)))

            elif snakeDict[0].x < 0 or snakeDict[0].y < 0 or snakeDict[0].x >= windowSize or snakeDict[0].y >= windowSize:
                learningValue = int(-2 * tilesWide)
                gameOver = True
                offScreen = True
                # print("DEBUG: Snake went out-of-bounds, last position was ", snakeDict[0].position)
            else:
                for i in range(1, len(snakeDict)):
                    if snakeDict[0].position == snakeDict[i].position:
                        learningValue = int(-2 * tilesWide)
                        gameOver = True
                        selfCollision = True
                        # print("DEBUG: Snake self-collision, snake segment number ", i)
                        break


    windowSize = int(800)
    tSize = int(windowSize / tilesWide) # tile size

    if snakeDict == None or apple == None:
        snakeDict = {}
        # snakeDict[0] is the snake's head
        snakeDict[0] = shape(randomCord(), randomCord(), tSize, tSize)

        apple = shape(0 - tSize, 0 - tSize, tSize, tSize)
        spawnApple()
    else:
        for i in range(0, len(snakeDict)):
            sSeg = snakeDict[i]
            snakeDict[i] = shape(sSeg.x, sSeg.y, sSeg.width, sSeg.height)
        
        oldA = apple
        apple = shape(oldA.x, oldA.y, oldA.width, oldA.height)

    handleInput() # run one move


    state = []
    for i in range(0, n_observations):
        state.append(0)
    state[0] = apple.x
    state[1] = apple.y

    maxObservableLength = int(n_observations / 2) - 1
    snakeLength = len(snakeDict)
    if snakeLength < maxObservableLength:
        observableLength = snakeLength
    else:
        observableLength = maxObservableLength

    for i in range(0, observableLength):
        state[(i + 1) * 2] = snakeDict[i].x
    for i in range(0, observableLength):
        state[(2 * i) + 3] = snakeDict[i].y

    if action == None:
        return state
    else:
        return state, learningValue, gameOver, snakeDict, apple, offScreen, selfCollision


if __name__ == '__main__':
    play()
