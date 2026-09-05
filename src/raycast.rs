use crate::{BaseFloat, ElementWise, InnerSpace, Point3, Vector3, Zero, bounding::{AABB3d, BoundingSphere, IntersectsVolume}};


#[derive(Clone, Debug)]
pub struct RayCast3d<S: BaseFloat> {
    pub origin: Point3<S>,
    pub direction: Vector3<S>,
    pub max: S,
    direction_reciprocal: Vector3<S>,
}

impl<S: BaseFloat + num_traits::Signed> RayCast3d<S> {
    /// Constructs a Raycast from an origin to a direction and max distance
    pub fn new(origin: Point3<S>, direction: Vector3<S>, max: S) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            direction_reciprocal: direction.normalize().reciprocal(),
            max,
        }
    }

    /// Gets the cached Direction Reciprocal
    pub const fn direction_recip(&self) -> Vector3<S> {
        self.direction_reciprocal
    }

    /// Gets the Intersection between some AABB and this RayCast and distance to said collision
    pub fn aabb_intersection_at(&self, aabb: &AABB3d<S>) -> Option<S> {
        let positive = self.direction.signum().gt(Vector3::zero());
        let min = Point3::select_vec(positive, aabb.min, aabb.max);
        let max = Point3::select_vec(positive, aabb.max, aabb.min);

        let tmin = (min - self.origin).mul_element_wise(self.direction_reciprocal);
        let tmax = (max - self.origin).mul_element_wise(self.direction_reciprocal);

        let tmin = tmin.max_element().max(S::from(0.0).unwrap());
        let tmax = tmax.min_element().min(S::from(self.max).unwrap());

        if tmin <= tmax {
            Some(tmin)
        } else {
            None
        }
    }

    pub fn sphere_intersection_at(&self, sphere: &BoundingSphere<S>) -> Option<S> {
        let offset = self.origin - sphere.center;
        let projected = offset.dot(self.direction);
        let closest_point = offset - self.direction * projected;
        let distance_squared = sphere.radius.powi(2) - closest_point.magnitude2();
        if distance_squared < S::from(0.0).unwrap() || projected.powi(2).copysign(-projected) < -distance_squared {
            None
        } else {
            let toi = -projected - distance_squared.sqrt();
            if toi > self.max {
                None
            } else {
                Some(toi.max(S::from(0.0).unwrap()))
            }
        }
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<AABB3d<S>> for RayCast3d<S> {
    fn intersects(&self, volume: &AABB3d<S>) -> bool {
        self.aabb_intersection_at(volume).is_some()
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<BoundingSphere<S>> for RayCast3d<S> {
    fn intersects(&self, volume: &BoundingSphere<S>) -> bool {
        self.sphere_intersection_at(volume).is_some()
    }
}

/// An interesection that casts an aabb3d along a ray
#[derive(Clone, Debug)]
pub struct AABBCast3d<S: BaseFloat> {
    pub ray: RayCast3d<S>,
    pub aabb: AABB3d<S>,
}

impl<S: BaseFloat + num_traits::Signed> AABBCast3d<S> {
    pub fn new(aabb: AABB3d<S>, origin: Point3<S>, direction: Vector3<S>, max: S) -> Self {
        Self {
            ray: RayCast3d::new(origin, direction, max),
            aabb
        }
    }

    pub fn aabb_collision_at(&self, mut aabb: AABB3d<S>) -> Option<S> {
        aabb.min.sub_assign_element_wise(self.aabb.max);
        aabb.max.sub_assign_element_wise(self.aabb.min);
        self.ray.aabb_intersection_at(&aabb)
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<AABB3d<S>> for AABBCast3d<S> {
    fn intersects(&self, volume: &AABB3d<S>) -> bool {
        self.aabb_collision_at(*volume).is_some()
    }
}

#[derive(Debug, Clone)]
pub struct BoundingSphereCast<S: BaseFloat> {
    pub ray: RayCast3d<S>,
    pub sphere: BoundingSphere<S>,
}

impl<S: BaseFloat + num_traits::Signed> BoundingSphereCast<S> {
    pub fn new(sphere: BoundingSphere<S>, origin: Point3<S>, direction: Vector3<S>, max: S) -> Self {
        Self {
            ray: RayCast3d::new(origin, direction, max),
            sphere,
        }
    }

    pub fn sphere_collision_at(&self, mut sphere: BoundingSphere<S>) -> Option<S> {
        sphere.center.sub_assign_element_wise(self.sphere.center);
        sphere.radius += self.sphere.radius;
        self.ray.sphere_intersection_at(&sphere)
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<BoundingSphere<S>> for BoundingSphereCast<S> {
    fn intersects(&self, volume: &BoundingSphere<S>) -> bool {
        self.sphere_collision_at(*volume).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Vector3, Point3};
    use crate::prelude::*;

    const EPSILON: f32 = 0.001;

    #[test]
    fn test_ray_intersection_sphere_hits() {
        for (test, volume, expected_distance) in &[
            (
                // Hit the center of a centered bounding sphere
                RayCast3d::new(Point3::new(0.0, 1.0, 0.0) * -5., Vector3::unit_y(), 90.),
                BoundingSphere::new(Point3::origin(), 1.),
                4.,
            ),
            (
                // Hit the center of a centered bounding sphere, but from the other side
                RayCast3d::new(Point3::new(0.0, 1.0, 0.0) * 5., -Vector3::unit_y(), 90.),
                BoundingSphere::new(Point3::origin(), 1.),
                4.,
            ),
            (
                // Hit the center of an offset sphere
                RayCast3d::new(Point3::origin(), Vector3::unit_y(), 90.),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 3., 2.),
                1.,
            ),
            (
                // Just barely hit the sphere before the max distance
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0), Vector3::unit_y(), 1.),
                BoundingSphere::new(Point3::new(1., 1., 0.), 0.01),
                0.99,
            ),
            (
                // Hit a sphere off-center
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0), Vector3::unit_y(), 90.),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 2.),
                3.268,
            ),
            (
                // Barely hit a sphere on the side
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0) * 0.99999, Vector3::unit_y(), 90.),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 1.),
                4.996,
            ),
        ] {
            assert!(
                test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
            let actual_distance = test.sphere_intersection_at(volume).unwrap();
            assert!(
                f32::abs(actual_distance - expected_distance) < EPSILON,
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}\n  Actual distance: {actual_distance}",
            );

            let inverted_ray = RayCast3d::new(test.origin, -test.direction, test.max);
            assert!(
                !inverted_ray.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
        }
    }

    #[test]
    fn test_ray_intersection_sphere_misses() {
        for (test, volume) in &[
            (
                // The ray doesn't go in the right direction
                RayCast3d::new(Point3::origin(), Vector3::unit_x(), 90.),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 2., 1.),
            ),
            (
                // Ray's alignment isn't enough to hit the sphere
                RayCast3d::new(Point3::origin(), Vector3::new(1., 1., 1.), 90.),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 2., 1.),
            ),
            (
                // The ray's maximum distance isn't high enough
                RayCast3d::new(Point3::origin(), Vector3::unit_y(), 0.5),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 2., 1.),
            ),
        ] {
            assert!(
                !test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}",
            );
        }
    }

    #[test]
    fn test_ray_intersection_sphere_inside() {
        let volume = BoundingSphere::new(Point3::from_value(0.5), 1.);
        for origin in &[Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0), Point3::from_value(1.0), Point3::origin()] {
            for direction in &[Vector3::unit_x(), Vector3::unit_y(), Vector3::unit_z(), -Vector3::unit_x(), -Vector3::unit_y(), -Vector3::unit_z()] {
                for max in &[0., 1., 900.] {
                    let test = RayCast3d::new(*origin, *direction, *max);

                    assert!(
                        test.intersects(&volume),
                        "Case:\n  origin: {origin:?}\n  Direction: {direction:?}\n  Max: {max}",
                    );

                    let actual_distance = test.sphere_intersection_at(&volume);
                    assert_eq!(
                        actual_distance,
                        Some(0.),
                        "Case:\n  origin: {origin:?}\n  Direction: {direction:?}\n  Max: {max}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_ray_intersection_aabb_hits() {
        for (test, volume, expected_distance) in &[
            (
                // Hit the center of a centered aabb
                RayCast3d::new(Point3::new(0.0, 1.0, 0.0) * -5., Vector3::unit_y(), 90.),
                AABB3d::new(Point3::origin(), Vector3::from_value(1.0)),
                4.,
            ),
            (
                // Hit the center of a centered aabb, but from the other side
                RayCast3d::new(Point3::new(0.0, 1.0, 0.0) * 5., -Vector3::unit_y(), 90.),
                AABB3d::new(Point3::origin(), Vector3::from_value(1.0)),
                4.,
            ),
            (
                // Hit the center of an offset aabb
                RayCast3d::new(Point3::origin(), Vector3::unit_y(), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 3., Vector3::from_value(2.)),
                1.,
            ),
            (
                // Just barely hit the aabb before the max distance
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0), Vector3::unit_y(), 1.),
                AABB3d::new(Point3::new(1., 1., 0.), Vector3::from_value(0.01)),
                0.99,
            ),
            (
                // Hit an aabb off-center
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0), Vector3::unit_y(), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 5., Vector3::from_value(2.)),
                3.,
            ),
            (
                // Barely hit an aabb on corner
                RayCast3d::new(Point3::new(1.0, 0.0, 0.0) * -0.001, Vector3::new(1., 1., 1.), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 2., Vector3::from_value(1.0)),
                1.732,
            ),
        ] {
            assert!(
                test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
            let actual_distance = test.aabb_intersection_at(volume).unwrap();
            assert!(
                f32::abs(actual_distance - expected_distance) < EPSILON,
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}\n  Actual distance: {actual_distance}",
            );

            let inverted_ray = RayCast3d::new(test.origin, -test.direction, test.max);
            assert!(
                !inverted_ray.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
        }
    }

    #[test]
    fn test_ray_intersection_aabb_misses() {
        for (test, volume) in &[
            (
                // The ray doesn't go in the right direction
                RayCast3d::new(Point3::origin(), Vector3::unit_x(), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 2., Vector3::from_value(1.0)),
            ),
            (
                // Ray's alignment isn't enough to hit the aabb
                RayCast3d::new(Point3::origin(), Vector3::new(1., 0.99, 1.), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 2., Vector3::from_value(1.0)),
            ),
            (
                // The ray's maximum distance isn't high enough
                RayCast3d::new(Point3::origin(), Vector3::unit_y(), 0.5),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 2., Vector3::from_value(1.0)),
            ),
        ] {
            assert!(
                !test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}",
            );
        }
    }

    #[test]
    fn test_ray_intersection_aabb_inside() {
        let volume = AABB3d::new(Point3::from_value(0.5), Vector3::from_value(1.0));
        for origin in &[Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0), Point3::from_value(1.0), Point3::origin()] {
            for direction in &[Vector3::unit_x(), Vector3::unit_y(), Vector3::unit_z(), -Vector3::unit_x(), -Vector3::unit_y(), -Vector3::unit_z()] {
                for max in &[0., 1., 900.] {
                    let test = RayCast3d::new(*origin, *direction, *max);

                    assert!(
                        test.intersects(&volume),
                        "Case:\n  origin: {origin:?}\n  Direction: {direction:?}\n  Max: {max}",
                    );

                    let actual_distance = test.aabb_intersection_at(&volume);
                    assert_eq!(
                        actual_distance,
                        Some(0.),
                        "Case:\n  origin: {origin:?}\n  Direction: {direction:?}\n  Max: {max}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_aabb_cast_hits() {
        for (test, volume, expected_distance) in &[
            (
                // Hit the center of the aabb, that a ray would've also hit
                AABBCast3d::new(AABB3d::new(Point3::origin(), Vector3::from_value(1.0)), Point3::origin(), Vector3::unit_y(), 90.),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 5., Vector3::from_value(1.0)),
                3.,
            ),
            (
                // Hit the center of the aabb, but from the other side
                AABBCast3d::new(
                    AABB3d::new(Point3::origin(), Vector3::from_value(1.0)),
                    Point3::new(0.0, 1.0, 0.0) * 10.,
                    -Vector3::unit_y(),
                    90.,
                ),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 5., Vector3::from_value(1.0)),
                3.,
            ),
            (
                // Hit the edge of the aabb, that a ray would've missed
                AABBCast3d::new(
                    AABB3d::new(Point3::origin(), Vector3::from_value(1.0)),
                    Point3::new(1.0, 0.0, 0.0) * 1.5,
                    Vector3::unit_y(),
                    90.,
                ),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 5., Vector3::from_value(1.0)),
                3.,
            ),
            (
                // Hit the edge of the aabb, by casting an off-center AABB
                AABBCast3d::new(
                    AABB3d::new(Point3::new(1.0, 0.0, 0.0) * -2., Vector3::from_value(1.0)),
                    Point3::new(1.0, 0.0, 0.0) * 3.,
                    Vector3::unit_y(),
                    90.,
                ),
                AABB3d::new(Point3::new(0.0, 1.0, 0.0) * 5., Vector3::from_value(1.0)),
                3.,
            ),
        ] {
            assert!(
                test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
            let actual_distance = test.aabb_collision_at(*volume).unwrap();
            assert!(
                f32::abs(actual_distance - expected_distance) < EPSILON,
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}\n  Actual distance: {actual_distance}",
            );

            let inverted_ray = RayCast3d::new(test.ray.origin, -test.ray.direction, test.ray.max);
            assert!(
                !inverted_ray.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
        }
    }

    #[test]
    fn test_sphere_cast_hits() {
        for (test, volume, expected_distance) in &[
            (
                // Hit the center of the bounding sphere, that a ray would've also hit
                BoundingSphereCast::new(
                    BoundingSphere::new(Point3::origin(), 1.),
                    Point3::origin(),
                    Vector3::unit_y(),
                    90.,
                ),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 1.),
                3.,
            ),
            (
                // Hit the center of the bounding sphere, but from the other side
                BoundingSphereCast::new(
                    BoundingSphere::new(Point3::origin(), 1.),
                    Point3::new(0.0, 1.0, 0.0) * 10.,
                    -Vector3::unit_y(),
                    90.,
                ),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 1.),
                3.,
            ),
            (
                // Hit the bounding sphere off-center, that a ray would've missed
                BoundingSphereCast::new(
                    BoundingSphere::new(Point3::origin(), 1.),
                    Point3::new(1.0, 0.0, 0.0) * 1.5,
                    Vector3::unit_y(),
                    90.,
                ),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 1.),
                3.677,
            ),
            (
                // Hit the bounding sphere off-center, by casting a sphere that is off-center
                BoundingSphereCast::new(
                    BoundingSphere::new(Point3::new(1.0, 0.0, 0.0) * -1.5, 1.),
                    Point3::new(1.0, 0.0, 0.0) * 3.,
                    Vector3::unit_y(),
                    90.,
                ),
                BoundingSphere::new(Point3::new(0.0, 1.0, 0.0) * 5., 1.),
                3.677,
            ),
        ] {
            assert!(
                test.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
            let actual_distance = test.sphere_collision_at(*volume).unwrap();
            assert!(
                f32::abs(actual_distance - expected_distance) < EPSILON,
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}\n  Actual distance: {actual_distance}",
            );

            let inverted_ray = RayCast3d::new(test.ray.origin, -test.ray.direction, test.ray.max);
            assert!(
                !inverted_ray.intersects(volume),
                "Case:\n  Test: {test:?}\n  Volume: {volume:?}\n  Expected distance: {expected_distance:?}",
            );
        }
    }
}
