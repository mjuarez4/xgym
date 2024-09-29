import xarm_gym as xg
from xarm_gym import utils as xgu

b = xgu.CartesianBoundary(
    xgu.PartialRobotState([1, 2, 3]), xgu.PartialRobotState([4, 5, 6])
    )
)
