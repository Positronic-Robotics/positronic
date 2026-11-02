# Changelog – YAM Description

All notable changes to this model will be documented in this file.

## [2026-11-02]

- Replace the model with the `arm/yam` and `gripper/linear_4310` files of i2rt at commit `5d47b35` (`PROVENANCE.md`).
  The model draws the linear gripper. The `gripper` site stays at its point on the arm. The base frame is 9.5 mm lower,
  at the bottom of the arm base.

## [2026-10-24]

- Replace the `default` site with the `gripper` site, on the axes of `link_6` at the centroid of the face where the
  closed fingertips meet in the i2rt CAD mesh. The driver reports poses there.

## [2026-08-03]

- Add the `default` site, coincident with `grasp_site`, as the frame the driver reports poses in.

## [2025-05-19]

- Initial release.
