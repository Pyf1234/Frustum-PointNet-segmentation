# Frustum-PointNet-segmentation
This is a Pytorch version of the Frustum  PointNet Segmentation
To simplify the implementation,I used the GT segmentation label to get the bounding box instead of object detection in the Paper.
Besides,I generate the point cloud by projecting all the pixels in the bounding box to 3D-xyz instead of projecting all the pixels in the frustum because I think the points outside the bounding box are useless to segmentation.
However you can easily implement original algorithm by modifying my code of box generation and point cloud generation.

