import copy
import random
import unittest
import json
from pathlib import Path

from snake.engine import Engine, MOVES
from tools.simulator import advance, initial, request, snake, spawn_food


class RulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def compare(self, state, actions):
        food, native = self.engine.transition(state, actions)
        python = advance(state, actions)
        self.assertEqual(sorted((p["x"],p["y"]) for p in food),
                         sorted((p["x"],p["y"]) for p in python["board"]["food"]))
        expected = {s["id"]: s for s in python["board"]["snakes"]}
        for actual in native:
            other = expected[actual["id"]]
            for key in ("health", "length", "alive"):
                self.assertEqual(actual[key], other[key], (key, actual, other))
            if all(p["x"] >= 0 and p["y"] >= 0 and p["x"] < state["board"]["width"] and p["y"] < state["board"]["height"] for p in other["body"]):
                self.assertEqual(actual["body"], other["body"])
        return native

    def test_equal_heads_die(self):
        state=request([snake("a",[(2,3),(1,3),(0,3)]),snake("b",[(4,3),(5,3),(6,3)])])
        out=self.compare(state,{"a":"right","b":"left"})
        self.assertEqual([s["alive"] for s in out],[False,False])

    def test_larger_head_wins(self):
        state=request([snake("a",[(2,3),(1,3),(0,3),(0,2)]),snake("b",[(4,3),(5,3),(6,3)])])
        out=self.compare(state,{"a":"right","b":"left"})
        self.assertEqual([s["alive"] for s in out],[True,False])

    def test_food_growth_duplicates_new_tail(self):
        state=request([snake("a",[(2,2),(2,1),(1,1)])],food=[(3,2)])
        out=self.compare(state,{"a":"right"})[0]
        self.assertEqual(out["body"], [{"x":3,"y":2},{"x":2,"y":2},{"x":2,"y":1},{"x":2,"y":1}])

    def test_food_restores_one_health_in_hazard(self):
        state=request([snake("a",[(2,2)]*3,1)],food=[(3,2)],hazards=[(3,2)])
        out=self.compare(state,{"a":"right"})[0]
        self.assertTrue(out["alive"]);self.assertEqual(out["health"],100)

    def test_hazard_stacks(self):
        state=request([snake("a",[(2,2)]*3,30)],hazards=[(3,2)]*2)
        self.assertEqual(self.compare(state,{"a":"right"})[0]["health"],1)

    def test_starvation(self):
        state=request([snake("a",[(2,2)]*3,1)])
        self.assertFalse(self.compare(state,{"a":"right"})[0]["alive"])

    def test_tail_vacates_even_when_eating(self):
        state=request([snake("a",[(1,1),(1,2),(2,2),(2,1)])],food=[(2,1)])
        self.assertTrue(self.compare(state,{"a":"right"})[0]["alive"])

    def test_duplicated_tail_does_not_vacate(self):
        state=request([snake("a",[(1,1),(1,2),(2,2),(2,1),(2,1)])])
        self.assertFalse(self.compare(state,{"a":"right"})[0]["alive"])

    def test_dead_body_from_wall_does_not_collide(self):
        state=request([snake("a",[(1,1),(2,1),(2,2)]),snake("b",[(0,1),(0,2),(0,3)])])
        out=self.compare(state,{"a":"left","b":"left"})
        self.assertEqual([s["alive"] for s in out],[True,False])

    def test_wrapped(self):
        state=request([snake("a",[(0,5),(1,5),(2,5)])],mode="wrapped")
        out=self.compare(state,{"a":"left"})[0]
        self.assertTrue(out["alive"]);self.assertEqual(out["head"],{"x":10,"y":5})

    def test_constrictor(self):
        state=request([snake("a",[(2,2)]*3)],mode="constrictor")
        out=self.compare(state,{"a":"right"})[0]
        self.assertEqual(out["length"],4);self.assertEqual(out["health"],100)

    def test_fuzz_transitions(self):
        rng=random.Random(91)
        for mode in ("standard","wrapped","constrictor"):
            for seed in range(50):
                state, game_rng=initial(seed,mode=mode)
                for _ in range(20):
                    actions={s["id"]:rng.choice(MOVES) for s in state["board"]["snakes"]}
                    self.compare(state,actions)
                    state=advance(state,actions)
                    if not all(s["alive"] for s in state["board"]["snakes"]):break
                    spawn_food(state,game_rng)


class TacticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.engine=Engine()

    def test_only_exit(self):
        state=request([snake("a",[(0,0),(0,1),(1,1),(2,1)])])
        self.assertEqual(self.engine.choose(state,50,4)[0],"right")

    def test_eat_or_starve(self):
        state=request([snake("a",[(2,2)]*3,1)],food=[(3,2)])
        self.assertEqual(self.engine.choose(state,50,4)[0],"right")

    def test_avoid_larger_head(self):
        state=request([snake("a",[(3,3),(3,2),(3,1)]),snake("b",[(5,3),(5,2),(5,1),(5,0)])],food=[(4,3)])
        move,_=self.engine.choose(state,50,4)
        self.assertNotEqual(move,"right")

    def test_finds_forced_kill(self):
        state=request([snake("a",[(2,0),(2,1),(2,2),(3,2)]),snake("b",[(0,0),(0,1),(0,2)])],width=4,height=4)
        move,stats=self.engine.choose(state,50,5)
        self.assertEqual(move,"left");self.assertGreater(stats["score"],900000)

    def test_deadline(self):
        state,_=initial(14)
        _,stats=self.engine.choose(state,20,20)
        self.assertLess(stats["engine_ms"],150)
        self.assertGreaterEqual(stats["depth"],1)

    def test_invalid_request_is_rejected(self):
        state,_=initial(14)
        state["board"]["width"]=1000
        with self.assertRaises(ValueError):self.engine.choose(state)

    def test_large_board_fallback(self):
        state=request([snake("a",[(1,1)]*3),snake("b",[(23,23)]*3)],
                      food=[(2,1),(12,12)],width=25,height=25)
        move,stats=self.engine.choose(state,100,3)
        self.assertIn(move,MOVES)
        self.assertEqual(stats["depth"],3)

    def test_long_tunnel_regression(self):
        positions=json.loads(Path(__file__).with_name("positions.json").read_text())
        # The 8 ms v1 entered a fatal tunnel here. A complete 10-ply search
        # must find the downward exit before the route closes.
        move,stats=self.engine.choose(positions[-1],350,10)
        self.assertEqual(stats["depth"],10)
        self.assertEqual(move,"down")

    def test_paired_seeds_preserve_board(self):
        a,_=initial(17,False);b,_=initial(17,True)
        self.assertEqual(a["board"]["food"],b["board"]["food"])
        self.assertEqual(a["board"]["snakes"][0]["body"],b["board"]["snakes"][1]["body"])


if __name__=="__main__":unittest.main()
