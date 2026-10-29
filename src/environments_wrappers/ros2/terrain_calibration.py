"""Explicitly enabled analytic calibration scene; never modifies natural rocks."""
import numpy as np

CENTER = np.array([-20., -20.])
ROOT = "/TerrainCalibration"


def fixture_spec():
    """Frozen development fixtures, all required boxes within 5m of start."""
    boxes = []
    for i, height in enumerate((.05, .08, .10, .14, .20)):
        for j, width in enumerate((.05, .10, .20)):
            boxes.append(dict(name=f"box_h{round(height*100):02}_w{round(width*100):02}",
                              center=[i-1.+.013, (1.5, 2.5, -1.5)[j]+.037],
                              width=width, height=height, required=height >= .08))
    return dict(center=CENTER.tolist(), boxes=boxes, floor_extent=6.,
                hole=dict(center=[-3.,-3.], width=.6),
                ramp=dict(center=[-3.5,2.], width=1., rise=.2),
                plateau=dict(center=[-3.,-1.5], width=1., height=.3),
                overhang=dict(center=[-3.,0.], width=1., bottom=.4, height=.1),
                reserved_route=[[0.,0.],[2.,0.]], development_fixture=True)


def create_scene(stage):
    from pxr import UsdGeom, UsdPhysics, Gf
    if stage.GetPrimAtPath(ROOT):
        raise ValueError("calibration prim already exists; refusing overwrite")
    UsdGeom.Xform.Define(stage, ROOT)
    spec = fixture_spec()

    def box(name, xy, size, bottom, height):
        cube = UsdGeom.Cube.Define(stage, ROOT+"/"+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(float(CENTER[0]+xy[0]), float(CENTER[1]+xy[1]), bottom+height/2))
        cube.AddScaleOp().Set(Gf.Vec3f(float(size[0]),float(size[1]),float(height)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Four non-overlapping slabs form a real unsupported hole. No hidden floor
    # beneath it and no ground clearing is performed in the natural scene.
    for name, bounds in [("floor_left",(-6.,-3.3,-6.,6.)),("floor_right",(-2.7,6.,-6.,6.)),
                         ("floor_bottom",(-3.3,-2.7,-6.,-3.3)),("floor_top",(-3.3,-2.7,-2.7,6.))]:
        x0,x1,y0,y1 = bounds
        box(name, ((x0+x1)/2,(y0+y1)/2), (x1-x0,y1-y0), -.2, .2)
    for item in spec["boxes"]:
        box(item["name"], item["center"], [item["width"]]*2, 0., item["height"])
    p, o, r = spec["plateau"], spec["overhang"], spec["ramp"]
    box("plateau", p["center"], [p["width"]]*2, 0., p["height"])
    box("overhang", o["center"], [o["width"]]*2, o["bottom"], o["height"])
    mesh = UsdGeom.Mesh.Define(stage, ROOT+"/ramp")
    x,y = CENTER+np.array(r["center"])
    mesh.CreatePointsAttr([(x-.5,y-.5,0.),(x+.5,y-.5,.2),(x+.5,y+.5,.2),(x-.5,y+.5,0.)])
    mesh.CreateFaceVertexCountsAttr([3,3])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,0,2,3])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return spec


def query_checks(spec):
    checks = []
    for b in spec["boxes"]:
        # Interior corners exercise the width, not only center/top height.
        for dx,dy in [(0.,0.),(-.4,-.4),(.4,.4)]:
            xy = CENTER+np.array(b["center"])+b["width"]*np.array([dx,dy])
            checks.append(dict(name=b["name"], xy=xy.tolist(), expected_z=b["height"]))
        xy = CENTER+np.array(b["center"])+np.array([b["width"]*.6,0.])
        checks.append(dict(name="outside_"+b["name"], xy=xy.tolist(), expected_z=0.))
    for name in ("plateau", "overhang"):
        item = spec[name]
        checks.append(dict(name=name, xy=(CENTER+item["center"]).tolist(),
                           expected_z=item.get("bottom",0.)+item["height"]))
    for offset in (-.4,0.,.4):
        xy = CENTER+np.array(spec["ramp"]["center"])+np.array([offset,0.])
        checks.append(dict(name="ramp", xy=xy.tolist(), expected_z=(offset+.5)*.2))
    checks.append(dict(name="hole", xy=(CENTER+spec["hole"]["center"]).tolist(), expected_z=None))
    # The geometric corridor is checked independently of mapper classifications.
    for x in np.linspace(-.8,2.8,19):
        for y in (-.7,0.,.7):
            checks.append(dict(name="corridor", xy=(CENTER+[x,y]).tolist(), expected_z=0.))
    return checks


def validate_queries(spec, cast):
    results = []
    for check in query_checks(spec):
        hits = cast((*check["xy"],2.), (0.,0.,-1.), 3.)
        # Robot occupies the start corridor; exclude only the robot identity.
        hits = [h for h in hits if not h[3].startswith("/Robots/")]
        actual = hits[0][1][2] if hits else None
        expected = check["expected_z"]
        passed = actual is None if expected is None else actual is not None and abs(actual-expected) <= .005
        results.append(dict(**check, actual_z=actual, passed=bool(passed)))
    return dict(passed=all(r["passed"] for r in results), checks=results, tolerance_m=.005,
                note="Analytic collider checks, not a movement clearance certificate or LiDAR recall gate.")
