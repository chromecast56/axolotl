from typing import Optional, List
from dataclasses import dataclass


class GPUCredits:
    def __init__(self):
        self.events = {}

        # a grant is two events -- an allocation, and a deallocation
        # a subtraction is one event

        self.start_to_end = {}


    def create_grant(self, grant_id: str, amount: int, expiration_timestamp: int, timestamp: int) -> None:
        self.start_to_end[timestamp] = expiration_timestamp

        self.events[timestamp] = (amount, grant_id)
        self.events[expiration_timestamp] = (-amount, grant_id)

        # when event expires, only remove existing amount
    
    def subtract(self, amount: int, timestamp: int) -> None:
        # when subtracting, greedily subtract from the grants that expire first (earliest expiration timestep)
        self.events[timestamp] = (-amount, None)

    def get_balance(self, timestamp: int) -> Optional[int]:
        if timestamp < 0:
            return None
        times = sorted(list(self.events.items()))
        avail_grants = {}
        res = 0
        for time, (amount, id) in times:
            if time > timestamp:
                break
            # allocate
            if id is not None and amount > 0:
                avail_grants[id] = (self.start_to_end[time], time, amount)
                res += avail_grants[id][2]
            # deallocate
            if id is not None and amount < 0:
                res -= avail_grants[id][2]
                del avail_grants[id]
            # subtract
            if id is None:
                amount = -amount
                res -= amount
                sorted_grants = sorted(avail_grants.items(), key=lambda x: x[1][0])
                # when subtracting, greedily subtract from the grants that expire first (earliest expiration timestep)
                for grant_id, (grant_exp_time, grant_time, grant_amount) in sorted_grants:
                    if amount >= grant_amount: # roll over
                        amount -= grant_amount
                        avail_grants[grant_id] = (grant_exp_time, grant_time, 0)
                    else: # stop here
                        avail_grants[grant_id] = (grant_exp_time, grant_time, grant_amount - amount)
                        amount = 0
                    if amount == 0:
                        break
                if amount > 0: # not enough credits :(
                    assert len(avail_grants) == 0
                    return None
            if res < 0:
                return None
        return res

if __name__ == "__main__":
    gpc = GPUCredits()
    gpc.subtract(amount=1, timestamp=30)
    assert gpc.get_balance(timestamp=30) is None
    gpc.create_grant(grant_id="a", amount=1, timestamp=10, expiration_timestamp=100)
    assert gpc.get_balance(timestamp=10) == 1
    assert gpc.get_balance(timestamp=20) == 1
    # print(gpc.get_balance(timestamp=30))
    assert gpc.get_balance(timestamp=30) == 0

    gpc = GPUCredits()
    gpc.subtract(amount=1, timestamp=30)
    assert gpc.get_balance(timestamp=30) is None
    gpc.create_grant(grant_id="a", amount=2, timestamp=10, expiration_timestamp=100)
    assert gpc.get_balance(timestamp=10) == 2
    assert gpc.get_balance(timestamp=20) == 2
    assert gpc.get_balance(timestamp=30) == 1
    # print(gpc.get_balance(timestamp=100))

    assert gpc.get_balance(timestamp=100) == 0

    gpc = GPUCredits()
    gpc.create_grant(grant_id="a", amount=3, timestamp=10, expiration_timestamp=60)
    assert gpc.get_balance(10) == 3
    gpc.create_grant(grant_id="b", amount=2, timestamp=20, expiration_timestamp=40)
    gpc.subtract(amount=1, timestamp=30)
    gpc.subtract(amount=3, timestamp=50)
    assert gpc.get_balance(10) == 3
    assert gpc.get_balance(20) == 5
    assert gpc.get_balance(30) == 4
    assert gpc.get_balance(40) == 3
    assert gpc.get_balance(50) == 0

    # out-of-order subtraction and future subtraction
    gpc = GPUCredits()
    gpc.create_grant(grant_id="a", amount=3, timestamp=10, expiration_timestamp=60)
    assert gpc.get_balance(10) == 3
    gpc.subtract(amount=3, timestamp=50)
    gpc.subtract(amount=10, timestamp=60)
    gpc.create_grant(grant_id="b", amount=2, timestamp=20, expiration_timestamp=40)
    gpc.subtract(amount=1, timestamp=30)

    assert gpc.get_balance(10) == 3
    assert gpc.get_balance(20) == 5
    assert gpc.get_balance(30) == 4
    assert gpc.get_balance(40) == 3
    assert gpc.get_balance(50) == 0



    gpc = GPUCredits()
    gpc.create_grant(grant_id="b", amount=2, timestamp=20, expiration_timestamp=40)
    gpc.create_grant(grant_id="a", amount=3, timestamp=10, expiration_timestamp=60)
    assert gpc.get_balance(10) == 3
    gpc.subtract(amount=3, timestamp=50)
    gpc.subtract(amount=10, timestamp=60)
    gpc.subtract(amount=1, timestamp=30)

    assert gpc.get_balance(10) == 3
    assert gpc.get_balance(20) == 5
    assert gpc.get_balance(30) == 4
    assert gpc.get_balance(40) == 3
    assert gpc.get_balance(50) == 0





