import asyncio
from sdk.commands.move_coordinates_command import MoveCoordinatesParamsPosition, MoveCoordinatesParamsOrientation, PlannerType
from sdk.commands.data import Point3D, Position, Orientation, JointPosition, JointPositions, FlangePinName, OutputType, VoltageLevel, VPWRType
from sdk.manipulators.medu import MEdu

host: str = '172.17.0.1'
client_id: str = '1227'
login: str = '13'
password: str = '14'

manipulator = MEdu(host, client_id, login, password)

async def make_manipulator_connection():
    try:
        print("Connect")
        manipulator.connect()
        print("get_control")
        manipulator.get_control()
        
        pos = MoveCoordinatesParamsPosition(1, 1, 1)
        orient = MoveCoordinatesParamsOrientation(1, 1, 1, 1, 1, 1, 1)
        manipulator.move_to_coordinates(pos, orient, 1, 1)
        
    except Exception as e:
        print('Возникла ошибка')
        print(e)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
asyncio.get_event_loop().run_until_complete(make_manipulator_connection())
