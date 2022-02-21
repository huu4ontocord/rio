
#use english firstnames for now
bantu_surnames = ["Dlamini", "Gumede", "Hadebe", "Ilunga", "Kamau", "Khoza", "Lubega", "M'Bala", "Mabaso", "Mabika",
                  "Mabizela", "Mabunda", "Mabuza", "Macharia", "Madima", "Madondo", "Mahlangu", "Maidza", "Makhanya",
                  "Malewezi", "Mamba", "Mandanda", "Mandlate", "Mangwana", "Manjate", "Maponyane", "Mapunda", "Maraire",
                  "Masango", "Maseko", "Masemola", "Masengo", "Mashabane", "Masire", "Masondo", "Masuku", "Mataka",
                  "Matovu", "Mbala", "Mbatha", "Mbugua", "Mchunu", "Mkhize", "Mofokeng", "Mokonyane", "Mutombo",
                  "Ncube", "Ndagire", "Ndhlovu", "Ndikumana", "Ndiritu", "Ndlovu", "Ndzinisa", "Ngcobo", "Nkomo",
                  "Nkosi", "Nkurunziza", "Radebe", "Tshabalala", "Tshivhumbe", "Vila"]

# male and female. Note some popular firstnames can be used for both male and female in Vietnamese
# vietnamese_firstnames = ["Anh", "Dung", "Hanh", "Hoa", "Hong", "Khanh", "Lan", "Liem", "Nhung", "Duy", "Xuan"]
# vietnamese_firstnames = ["Trọng", "Hiền", "Bích", "Duy", "Yến", "Sơn", "Oanh", "Mẫn", "Đông", "Lam", "Tín", "Nhi", "Phúc", "Ngọc", "Dương", "Loan", "Luân", "Nhàn", "Hiếu", "Băng", "Phát", "Chi", "Thịnh", "Di", "Tuấn", "Vi", "Hoa", "Diệu", "Tú", "Uyên", "Huy", "Văn", "Kỳ", "Thy", "Như", "Hằng", "Thông", "Minh", "Xuân", "Diệp", "Thúy", "Trang", "Hòa", "Trinh", "An", "Khang", "Thành", "Thiên", "Khôi", "Trâm", "Khiêm",
#                          "Nam", "Tân", "Hân", "Hương", "Hồng", "Vỹ", "Phụng", "Sang", "Mạnh", "Ý", "Bách", "Long", "Kha", "Phi", "Châu", "Trúc", "Kiệt", "Nhựt", "Thái", "Trường", "Duyên", "Diễm", "Vy", "Quỳnh", "Linh", "Giang", "Huyền", "Khanh", "Quân", "Hoàng", "Tiến", "Dũng", "Cường", "Vinh", "Hiển", "Hạnh", "Tùng", "Đại", "Quốc", "Bình", "Tâm", "Tuyết", "Lợi", "Phương", "Mỹ", "Liên", "Đức", "Vương", "Thi", "Công",
#                          "Thanh", "Toàn", "Hải", "Triết", "Thiện", "Thùy", "Lâm", "Anh", "Tài", "Thư", "Đạt", "Tuyền", "Đăng", "Hưng", "Khải", "Hào", "Lan", "Trung", "Trí", "Ly", "Tường", "Nga", "Phước", "Hậu", "Tiên", "Nhật", "Ngân", "Khuê", "Thắng", "Ánh", "Thơ", "Phong", "Nhung", "Hiệp", "Thủy", "Doanh", "Phú", "Mai", "Nhã", "Hùng", "Việt", "Mi", "Kiên", "Kiều", "Thuận", "Nhiên", "Dung", "Trà", "Thắm", "Đan", "Thương",
#                          "Đào", "Nguyệt", "Trân", "Khánh", "Kim", "Tuệ", "Nguyên", "Vĩ", "My", "Thảo", "Nghi", "Vũ", "Khoa", "Ân", "Danh", "Hà", "Phượng", "Quý", "Quang", "Quyên", "Vân", "Lộc", "Nhân", "Tấn", "Nghĩa", "Bảo"]
vietnamese_firstnames_female = ["Anh", "Vy", "Ngọc", "Nhi", "Hân", "Thư", "Linh", "Như", "Ngân", "Phương", "Thảo", "My", "Trân", "Quỳnh", "Nghi", "Trang", "Trâm", "An", "Thy", "Châu", "Trúc", "Uyên", "Yến", "Ý", "Tiên", "Mai", "Hà", "Vân", "Nguyên", "Hương", "Quyên", "Duyên", "Kim", "Trinh", "Thanh", "Tuyền", "Hằng", "Dương", "Chi", "Giang", "Tâm", "Lam", "Tú", "Ánh", "Hiền", "Khánh", "Minh", "Huyền", "Thùy", "Vi",
                              "Ly", "Dung", "Nhung", "Phúc", "Lan", "Phụng", "Ân", "Thi", "Khanh", "Kỳ", "Nga", "Tường", "Thúy", "Mỹ", "Hoa", "Tuyết", "Lâm", "Thủy", "Đan", "Hạnh", "Xuân", "Oanh", "Mẫn", "Khuê", "Diệp", "Thương", "Nhiên", "Băng", "Hồng", "Bình", "Loan", "Thơ", "Phượng", "Mi", "Nhã", "Nguyệt", "Bích", "Đào", "Diễm", "Kiều", "Hiếu", "Di", "Liên", "Trà", "Tuệ", "Thắm", "Diệu", "Quân", "Nhàn", "Doanh"]
vietnamese_firstnames_male = ["Huy", "Khang", "Bảo", "Minh", "Phúc", "Anh", "Khoa", "Phát", "Đạt", "Khôi", "Long", "Nam", "Duy", "Quân", "Kiệt", "Thịnh", "Tuấn", "Hưng", "Hoàng", "Hiếu", "Nhân", "Trí", "Tài", "Phong", "Nguyên", "An", "Phú", "Thành", "Đức", "Dũng", "Lộc", "Khánh", "Vinh", "Tiến", "Nghĩa", "Thiện", "Hào", "Hải", "Đăng", "Quang", "Lâm", "Nhật", "Trung", "Thắng", "Tú", "Hùng", "Tâm", "Sang", "Sơn", "Thái",
                               "Cường", "Vũ", "Toàn", "Ân", "Thuận", "Bình", "Trường", "Danh", "Kiên", "Phước", "Thiên", "Tân", "Việt", "Khải", "Tín", "Dương", "Tùng", "Quý", "Hậu", "Trọng", "Triết", "Luân", "Phương", "Quốc", "Thông", "Khiêm", "Hòa", "Thanh", "Tường", "Kha", "Vỹ", "Bách", "Khanh", "Mạnh", "Lợi", "Đại", "Hiệp", "Đông", "Nhựt", "Giang", "Kỳ", "Phi", "Tấn", "Văn", "Vương", "Công", "Hiển", "Linh", "Ngọc", "Vĩ"]
                         # source female: https://kiencang.net/100-ten-nu-gioi-pho-bien/, male: https://kiencang.net/100-ten-nam-pho-bien-vietnam/
vietnamese_surnames = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]

#use english firstnames for now
bengali_surnames  = ["Banerjee", "Bagchi", "Bhaduri", "Bhattacharjee", "Chakraborty", "Chatterjee", "Ganguly", "Goswami", "Ghoshal", "Lahiri", "Maitra", "Mukherjee", "Sanyal", "Chakraborty", "Bhattacharya", "Baidya", "Sengupta", "Dasgupta", "Duttagupta", "Gupta", "Sen-Sharma", "Basu", "Bose", "Dutta", "Ghosh", "Choudhury", "Guha", "Gain", "Mitra", "Singh", "Sinha", "Sen", "Pal", "De", "Dey", "Deb", "Dev", "Jana", "Palit", "Chanda", "Chandra", "Das", "Dam", "Kar", "Nandi", "Sarkar", "Nag", "Som"]

#male and female
urdu_firstnames = ["Azhar", "Benazir", "Farahnaz", "Maliha", "Minoo", "Romana", "Sania", "Azhar", "Burhan", "Changezi", "Faizan", "Fasih", "Fuad", "Hassim", "Jan", "Shoaib"]
urdu_surnames = ["Abid", "Ahmad", "Akbar", "Akmal", "Alam", "Ayaz", "Bohra", "Bucha", "Bukhari", "Buksh", "Bux", "Chandpuri", "Changezi", "Emani", "Farrukhabadi", "Farrukhi", "Fazail", "Hassim", "Hilaly", "Hussaini ", "Brahmin", "Lucknawi", "Ludhianvi", "Matloob", "Omar", "Vaishya", "Rahimtoola", "Shafiq", "Shoaib", "Siddiqui", "Siddiqui", "Tikka", "Yasser", ]

#basque and catalan - use Spanish names for now
