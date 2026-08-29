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
            direction,
            direction_reciprocal: direction.reciprocal(),
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
        let tmax = tmax.min_element().max(S::from(self.max).unwrap());

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
