# custom gym envs

### important global environment parameters:
```
osu!
score = frame[0:30, 675:800, :]
pyautogui.moveto(960, 535) # for reset()

McOsu:
score = frame[13:40, 650:800, :]
pyautogui.moveTo(990, 535) # for reset()
```

**NOTE: move to a global config file or something later**