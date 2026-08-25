use std::f32::consts;

use crate::{
    BaseFloat, ElementWise, EuclideanSpace, InnerSpace, Matrix3, MetricSpace, Point3, Quaternion, Rotation, Transform, Vector3, Zero,
};

/// A trait that generalizes different bounding volumes.
/// Bounding volumes are simplified shapes that are used to get simpler ways to check for
/// overlapping elements or finding intersections.
///
/// This trait supports both 2D and 3D bounding shapes.
pub trait BoundingVolume: Sized {
    /// The position type used for the volume. This should be `Vec2` for 2D and `Vec3` for 3D.
    type Translation: Clone + Copy + PartialEq;

    /// The rotation type used for the volume. This should be `Rot2` for 2D and `Quat` for 3D.
    type Rotation: Clone + Copy + PartialEq;

    /// The type used for the size of the bounding volume. Usually a half size. For example an
    /// `f32` radius for a circle, or a `Vec3` with half sizes for x, y and z for a 3D axis-aligned
    /// bounding box
    type HalfSize;

    /// The Type used for the size of the Visible Area (since Bounding volume is implented for Types that have that Generic this will have to respect that)
    type S: BaseFloat;

    /// Returns the center of the bounding volume.
    fn center(&self) -> Self::Translation;

    /// Returns the half size of the bounding volume.
    fn half_size(&self) -> Self::HalfSize;

    /// Computes the visible surface area of the bounding volume.
    /// This method can be useful to make decisions about merging bounding volumes,
    /// using a Surface Area Heuristic.
    ///
    /// For 2D shapes this would simply be the area of the shape.
    /// For 3D shapes this would usually be half the area of the shape.
    fn visible_area(&self) -> Self::S;

    /// Checks if this bounding volume contains another one.
    fn contains(&self, other: &Self) -> bool;

    /// Computes the smallest bounding volume that contains both `self` and `other`.
    fn merge(&self, other: &Self) -> Self;

    /// Increases the size of the bounding volume in each direction by the given amount.
    fn grow(&self, amount: Self::HalfSize) -> Self;

    /// Decreases the size of the bounding volume in each direction by the given amount.
    fn shrink(&self, amount: Self::HalfSize) -> Self;

    /// Scale the size of the bounding volume around its center by the given amount
    fn scale_around_center(&self, scale: Self::HalfSize) -> Self;

    /// Transforms the bounding volume by first rotating it around the origin and then applying a translation.
    fn transformed_by(
        mut self,
        translation: impl Into<Self::Translation>,
        rotation: impl Into<Self::Rotation>
    ) -> Self {
        self.transform_by(translation, rotation);
        self
    }

    /// Transforms the bounding volume by first rotating it around the origin and then applying a translation.
    fn transform_by(
        &mut self,
        translation: impl Into<Self::Translation>,
        rotation: impl Into<Self::Rotation>
    ) {
        self.rotate_by(rotation);
        self.translate_by(translation);
    }

    /// Translates the bounding volume by the given translation.
    fn translated_by(mut self, translation: impl Into<Self::Translation>) -> Self {
        self.translate_by(translation);
        self
    }

    /// Translates the bounding volume by the given translation.
    fn translate_by(&mut self, translation: impl Into<Self::Translation>);

    /// Rotates the bounding volume around the origin by the given rotation.
    ///
    /// The result is a combination of the original volume and the rotated volume,
    /// so it is guaranteed to be either the same size or larger than the original.
    fn rotated_by(mut self, rotation: impl Into<Self::Rotation>) -> Self {
        self.rotate_by(rotation);
        self
    }

    /// Rotates the bounding volume around the origin by the given rotation.
    ///
    /// The result is a combination of the original volume and the rotated volume,
    /// so it is guaranteed to be either the same size or larger than the original.
    fn rotate_by(&mut self, rotation: impl Into<Self::Rotation>);
}

/// A trait that generalizes intersection tests against a volume.
/// Intersection tests can be used for a variety of tasks, for example:
/// - Raycasting
/// - Testing for overlap
/// - Checking if an object is within the view frustum of a camera
pub trait IntersectsVolume<Volume: BoundingVolume> {
    /// Check if a volume intersects with this intersection test
    fn intersects(&self, volume: &Volume) -> bool;
}

// ========== 3D ==========

/// A 3D Axis Aligned bounding Box
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AABB3d<S: BaseFloat> {
    /// The minimum point of the box
    pub min: Point3<S>,
    /// The Maximum point of the box
    pub max: Point3<S>,
}

impl<S: BaseFloat + num_traits::Signed> AABB3d<S> {
    /// Constructs AABB from its center and half size
    pub fn new(center: Point3<S>, half_size: Vector3<S>) -> Self {
        Self { min: center - half_size, max: center + half_size }
    }

    /// constructs an AABB from its minimum and maximum extent
    pub fn from_min_max(min: Point3<S>, max: Point3<S>) -> Self {
        Self { min, max }
    }

    // Computes the smallest [`BoundingSphere`] containing this [`Aabb3d`]
    pub fn bounding_sphere(&self) -> BoundingSphere<S> {
        let radius = self.min.distance(self.max) / S::from(2.0).unwrap();
        BoundingSphere::new(self.center(), radius)
    }

    pub fn closest_point(&self, point: Point3<S>) -> Point3<S> {
        point.clamp(self.min, self.max)
    }
}

impl<S: BaseFloat + num_traits::Signed> BoundingVolume for AABB3d<S> {
    type Translation = Point3<S>;

    type Rotation = Quaternion<S>;

    type HalfSize = Vector3<S>;

    type S = S;

    fn center(&self) -> Self::Translation {
        self.min.midpoint(self.max) / S::from(2.0).unwrap()
    }

    fn half_size(&self) -> Self::HalfSize {
        (self.max - self.min) / S::from(2.0).unwrap()
    }

    fn visible_area(&self) -> S {
        let b = (self.max - self.min).max(Vector3::zero());
        b.x * (b.y + b.z) + b.y * b.z
    }

    fn contains(&self, other: &Self) -> bool {
        other.min.all_ge(self.min) && other.max.all_le(self.max)
    }

    fn merge(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn grow(&self, amount: Self::HalfSize) -> Self {
        let b = Self {
            min: self.min - amount,
            max: self.max + amount,
        };
        b
    }

    fn shrink(&self, amount: Self::HalfSize) -> Self {
        let b = Self {
            min: self.min + amount,
            max: self.max - amount,
        };
        b
    }

    fn scale_around_center(&self, scale: Self::HalfSize) -> Self {
        let b = Self {
            min: self.center() - self.half_size().mul_element_wise(scale),
            max: self.center() + self.half_size().mul_element_wise(scale),
        };
        b
    }

    fn translate_by(&mut self, translation: impl Into<Self::Translation>) {
        let translation = translation.into();
        self.min += translation.to_vec();
        self.max += translation.to_vec();
    }

    fn rotate_by(&mut self, rotation: impl Into<Self::Rotation>) {
        let rot_mat = Matrix3::from(rotation.into());
        let half_size = rot_mat.abs() * self.half_size();
        *self = Self::new(rot_mat.transform_point(self.center()), half_size);
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<Self> for AABB3d<S> {
    fn intersects(&self, volume: &Self) -> bool {
        self.min.all_le(volume.max) && self.max.all_ge(volume.min)
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<BoundingSphere<S>> for AABB3d<S> {
    fn intersects(&self, volume: &BoundingSphere<S>) -> bool {
        let closest_point = self.closest_point(volume.center);
        let distance_squared = volume.center.distance2(closest_point);
        let radius_squared = volume.radius.powi(2);
        distance_squared <= radius_squared
    }
}

/// A Bounding Sphere
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundingSphere<S: BaseFloat> {
    pub center: Point3<S>,
    pub radius: S,
}

impl<S: BaseFloat> BoundingSphere<S> {
    /// Constructs a bounding sphere from its center and radius
    pub fn new(center: Point3<S>, radius: S) -> Self {
        Self { center, radius }
    }

    /// Computes the smallest [`AABB3d`] containing this Bounding sphere
    pub fn aabb_3d(&self) -> AABB3d<S> {
        AABB3d {
            max: self.center.add_element_wise(self.radius),
            min: self.center.add_element_wise(self.radius),
        }
    }

    /// Finds the point on the bounding sphere that is closest to the given `point`
    ///
    /// If the point is outside the sphere the returned point will be on the surface of the sphere
    /// Otherwise, it will be inside the sphere and returned as is.
    pub fn closest_point(&self, point: Point3<S>) -> Point3<S> {
        let distance_squared = self.center.distance2(point);

        if distance_squared <= self.radius.powi(2) {
            point
        } else {
            let dir_to_point = (point / distance_squared.sqrt()).to_vec();
            self.center + dir_to_point * self.radius
        }
    }
}

impl<S: BaseFloat + num_traits::Signed> BoundingVolume for BoundingSphere<S> {
    type Translation = Vector3<S>;

    type Rotation = Quaternion<S>;

    type HalfSize = S;

    type S = S;

    fn center(&self) -> Self::Translation {
        self.center.to_vec()
    }

    fn half_size(&self) -> Self::HalfSize {
        self.radius
    }

    fn visible_area(&self) -> Self::S {
        S::from(2.0).unwrap() * S::from(consts::PI).unwrap() * self.radius * self.radius
    }

    fn contains(&self, other: &Self) -> bool {
        let diff = self.radius - other.radius;
        self.center().distance2(other.center.to_vec()) <= diff.powi(2).copysign(diff)
    }

    fn merge(&self, other: &Self) -> Self {
        let diff = other.center - self.center;
        let length = diff.magnitude();
        if self.radius >= length + other.radius {
            return *self;
        }
        if other.radius >= length + self.radius {
            return *other;
        }
        let dir = diff / length;
        Self::new(self.center.midpoint(other.center) + dir * ((other.radius - self.radius) / S::from(2.0).unwrap()),
            (length + self.radius + other.radius) / S::from(2.0).unwrap()
        )
    }

    fn grow(&self, amount: Self::HalfSize) -> Self {
        Self {
            center: self.center,
            radius: self.radius + amount,
        }
    }

    fn shrink(&self, amount: Self::HalfSize) -> Self {
        Self {
            center: self.center,
            radius: self.radius - amount,
        }
    }

    fn scale_around_center(&self, scale: Self::HalfSize) -> Self {
        Self::new(self.center, self.radius * scale)
    }

    fn translate_by(&mut self, translation: impl Into<Self::Translation>) {
        self.center += translation.into();
    }

    fn rotate_by(&mut self, rotation: impl Into<Self::Rotation>) {
        let rotation = rotation.into();
        self.center = rotation.rotate_point(self.center);
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<Self> for BoundingSphere<S> {
    fn intersects(&self, volume: &Self) -> bool {
        let center_distance_squared = self.center.distance2(volume.center);
        let radius_sum_squared = (self.radius + volume.radius).powi(2);
        center_distance_squared <= radius_sum_squared
    }
}

impl<S: BaseFloat + num_traits::Signed> IntersectsVolume<AABB3d<S>> for BoundingSphere<S> {
    fn intersects(&self, volume: &AABB3d<S>) -> bool {
        volume.intersects(self)
    }
}
