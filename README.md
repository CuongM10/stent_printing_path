# Stent Graph Optimizer V2 — Straight Euler / Open CPP 



- `inputs/square_list.txt`
- `inputs/arowhead_list.txt`
- `inputs/honeycomb_list.txt`
- `inputs/reentrant_list'



Open CPP dùng hai dummy endpoints để tự chọn Start/End. Objective augmentation theo thứ tự ưu tiên:

1. ít **duplicate edge traversals** nhất;
2. nếu bằng nhau, ít **duplicate geometric length** nhất;
3. retract luôn bằng 0.

Sau augmentation, chương trình sinh nhiều Euler candidates và xếp hạng theo support rồi độ mượt.

## Support score

Các bước được phân loại:



Thứ tự mong muốn: `STRONG_INTERSECTION > CONTACT_SUPPORT > UNSUPPORTED`.


- `optimization_summary.json`
- `optimized_route.csv`
- `transition_report.csv`
- `route_node_sequence.txt`
- `optimized_route_2d.png` — có node + mũi tên + thứ tự màu
- `support_report.txt`
- `ocpp_augmentation.json`
- `duplicated_edges_route.csv`
- `toolpath_rotary_AZ.gcode`
- `viewer_3d_cylinder.html` — Plotly embedded, mở offline

### Ký hiệu trên hình 2D

- màu: thứ tự in;
- mũi tên: hướng chạy;
- vòng đỏ: `avoidable interior bend`;
- đường nét đứt đen: OCPP duplicate traversal.

