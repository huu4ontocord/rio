
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

vietnamese_firstnames_female = ["Anh", "Vy", "Ngọc", "Nhi", "Hân", "Thư", "Linh", "Như", "Ngân", "Phương", "Thảo", "My", "Trân", "Quỳnh", "Nghi", "Trang", "Trâm", "An", "Thy", "Châu", "Trúc", "Uyên", "Yến", "Ý", "Tiên", "Mai", "Hà", "Vân", "Nguyên", "Hương", "Quyên", "Duyên", "Kim", "Trinh", "Thanh", "Tuyền", "Hằng", "Dương", "Chi", "Giang", "Tâm", "Lam", "Tú", "Ánh", "Hiền", "Khánh", "Minh", "Huyền", "Thùy", "Vi",
                              "Ly", "Dung", "Nhung", "Phúc", "Lan", "Phụng", "Ân", "Thi", "Khanh", "Kỳ", "Nga", "Tường", "Thúy", "Mỹ", "Hoa", "Tuyết", "Lâm", "Thủy", "Đan", "Hạnh", "Xuân", "Oanh", "Mẫn", "Khuê", "Diệp", "Thương", "Nhiên", "Băng", "Hồng", "Bình", "Loan", "Thơ", "Phượng", "Mi", "Nhã", "Nguyệt", "Bích", "Đào", "Diễm", "Kiều", "Hiếu", "Di", "Liên", "Trà", "Tuệ", "Thắm", "Diệu", "Quân", "Nhàn", "Doanh"]
vietnamese_firstnames_male = ["Huy", "Khang", "Bảo", "Minh", "Phúc", "Anh", "Khoa", "Phát", "Đạt", "Khôi", "Long", "Nam", "Duy", "Quân", "Kiệt", "Thịnh", "Tuấn", "Hưng", "Hoàng", "Hiếu", "Nhân", "Trí", "Tài", "Phong", "Nguyên", "An", "Phú", "Thành", "Đức", "Dũng", "Lộc", "Khánh", "Vinh", "Tiến", "Nghĩa", "Thiện", "Hào", "Hải", "Đăng", "Quang", "Lâm", "Nhật", "Trung", "Thắng", "Tú", "Hùng", "Tâm", "Sang", "Sơn", "Thái",
                               "Cường", "Vũ", "Toàn", "Ân", "Thuận", "Bình", "Trường", "Danh", "Kiên", "Phước", "Thiên", "Tân", "Việt", "Khải", "Tín", "Dương", "Tùng", "Quý", "Hậu", "Trọng", "Triết", "Luân", "Phương", "Quốc", "Thông", "Khiêm", "Hòa", "Thanh", "Tường", "Kha", "Vỹ", "Bách", "Khanh", "Mạnh", "Lợi", "Đại", "Hiệp", "Đông", "Nhựt", "Giang", "Kỳ", "Phi", "Tấn", "Văn", "Vương", "Công", "Hiển", "Linh", "Ngọc", "Vĩ"]
vietnamese_first_middlenames_male = ["Minh", "Hoàng", "Gia", "Nguyễn", "Quốc", "Thanh", "Văn", "Thành", "Anh", "Ngọc", "Tấn", "Đức", "Lê", "Tuấn", "Quang", "Trần", "Hữu", "Nhật", "Duy", "Trọng", "Đình", "Đăng", "Huỳnh", "Trung", "Bảo", "Phúc", "Tiến", "Chi", "Thiên", "Công", "Xuân", "Phạm", "Vũ", "Thái", "Huy", "Võ", "Hải", "Thế", "Hồng", "Khánh", "Tri", "Phước", "Phú", "Nguyên", "Việt", "Mạnh", "Bá", "Trường", "Vĩnh", "Hoài",
                                    "Phan", "Cao", "Đặng", "Hồ", "Dương", "Thiện", "Lâm", "Kim", "Đỗ", "Trương", "Đại", "Viết", "Phi", "Phương", "Nam", "Đoàn", "Hà", "Kiến", "Ngô", "Nhựt", "Hiếu", "Bùi", "An", "Hùng", "Chấn", "Bình", "Khải", "Khắc", "Khôi", "Mai", "Châu", "Sỹ", "Vĩ", "Tùng", "Lý", "Long", "Hưng", "Hạo", "Phát", "Như", "Đinh", "Quý", "Đắc", "Vinh", "Nhất", "Đông", "Lương", "Kỳ", "Trịnh", "Thuận"]
vietnamese_second_middlenames_male = ["Minh", "Gia", "Anh", "Hoàng", "Quốc", "Bảo", "Tuấn", "Thiên", "Đăng", "Thanh", "Nhật", "Thành", "Duy", "Tấn", "Đức", "Phúc", "Quang", "Khánh", "Trung", "Hải", "Ngọc", "Trọng", "Huy", "Thái", "Hữu", "Tiến", "Nguyên", "Trường", "Tri", "Phú", "Phước", "Hoài", "An", "Nam", "Việt", "Phương", "Xuân", "Chi", "Thế", "Phi", "Khôi", "Công", "Thiện", "Hồng", "Vĩnh", "Bình", "Đình", "Đại", "Lê", "Mạnh",
                                    "Hiếu", "Văn", "Nhựt", "Kim", "Vũ", "Kỳ", "Long", "Bá", "Đông", "Hùng", "Hưng", "Khang", "Cao", "Kiến", "Sơn", "Nhất", "Tùng", "Phát", "Lâm", "Khải", "Thuận", "Tâm", "Hạo", "Nhân", "Triệu", "Vinh", "Chấn", "Tường", "Phong", "Quý", "Nguyễn", "Như", "Huỳnh", "Song", "Thịnh", "Triều", "Châu", "Vương", "Tuần", "Sỹ", "Tài", "Hà", "Hoàn", "Khắc", "Linh", "Toàn", "Tần", "Viết", "Hà", "Bách"]
vietnamese_first_middlenames_female = ["Thị", "Ngọc", "Nguyễn", "Hoàng", "Lê", "Trần", "Thanh", "Bảo", "Phương", "Huỳnh", "Gia", "Minh", "Kim", "Quỳnh", "Phạm", "Khánh", "Hòng", "Mỹ", "Hà", "Vũ", "Võ", "Mai", "Thùy", "Anh", "Như", "Thảo", "Thụy", "Phan", "Yến", "Đặng", "Xuân", "Hồ", "Thiên", "Đỗ", "Nhật", "Thái", "Tường", "Tuyết", "Nhã", "Thúy", "Dương", "Hải", "Thu", "Lâm", "Trúc", "Trương", "Hoài", "Đoàn", "Ngô", "Tú",
                                      "Cao", "Kiều", "Khánh", "Phúc", "Bích", "Châu", "Bùi", "Khả", "Vân", "Đình", "Tâm", "Thục", "Bội", "Ái", "Lý", "Hương", "Nguyên", "Uyên", "Thủy", "Trịnh", "Cẩm", "Đào", "Diệp", "Tuệ", "Diệu", "Huệ", "Diễm", "Lan", "Cát", "Huyền", "An", "Linh", "Lưu", "Quế", "Ngân", "Đinh", "Uyển", "Triệu", "Trà", "Song", "Bình", "Nguyệt", "Trang", "Mẫn", "Kỳ", "Trâm", "Hạnh", "Lương", "Vương", "Tiểu"]
vietnamese_second_middlenames_female = ["Bảo", "Ngọc", "Phương", "Thanh", "Minh", "Kim", "Quỳnh", "Khánh", "Như", "Thảo", "Anh", "Yến", "Gia", "Mỹ", "Thùy", "Hồng", "Tường", "Thiên", "Hoàng", "Thu", "Tuyết", "Trúc", "Mai", "Xuân", "Thúy", "Bích", "Cẩm", "Ánh", "Kiều", "Diễm", "Hà", "Lan", "Hải", "Thủy", "Nhã", "Vân", "Trâm", "Trà", "Tú", "Cát", "Uyên", "Hoài", "Huyền", "Huỳnh", "Linh", "Nhật", "Hương", "Tâm", "An", "Diệu",
                                        "Ái", "Ngân", "Đan", "Khả", "Kỳ", "Thị", "Quế", "Tố", "Đông", "Thái", "Song", "Nam", "Phi", "Hạnh", "Ý", "Thục", "Phúc", "Châu", "Tuệ", "Uyển", "Nguyệt", "Đoan", "Lê", "Nguyên", "Mộng", "Bình", "Trang", "Lam", "Hiền", "Băng", "Mẫn", "Thụy", "Vy", "Hạ", "Việt", "Hiếu", "Triệu", "Trường", "Lệ", "Phượng", "Diệp", "Lâm", "Thy", "Bé", "Yên", "Khải", "Tiểu", "Huệ", "Phước", "Đỗ"]
vietnamese_surnames = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]

#use english firstnames for now
bengali_surnames  = ['Bagchi', 'Baidya', 'Banerjee', 'Basu', 'Bhaduri', 
	'Bhattacharjee', 'Bhattacharya', 'Bose', 'Chakraborty', 'Chakraborty', 
	'Chanda', 'Chandra', 'Chatterjee', 'Choudhury', 'Dam', 'Das', 'Das-Sharma', 
	'Dasgupta', 'De', 'Deb', 'Dev', 'Dey', 'Dutta', 'Duttagupta', 'Gain', 
	'Ganguly', 'Ghosh', 'Ghoshal', 'Goswami', 'Guha', 'Gupta', 'Jana', 
	'Kanjilal', 'Kar', 'Kumar', 'Kundu', 'Lahiri', 'Maitra', 'Mitra', 
	'Mukherjee', 'Nag', 'Nandi', 'Pal', 'Palit', 'Ray', 'Roy', 'Saha', 'Sanyal',
	'Sarkar', 'Sen', 'Sen-Sharma', 'Sengupta', 'Singh', 'Sinha', 'Som', 
	'কর', 'কানজিলাল', 'কুন্ডু', 'কুমার', 'গাঙ্গুলি', 'গুপ্ত', 'গুহ', 'গোস্বামী', 
	'ঘোষ', 'ঘোষাল', 'চক্রবর্তী', 'চক্রবর্তী', 'চন্দা', 'চন্দ্র', 'চৌধুরী', 
	'চ্যাটার্জি', 'জানা', 'দত্ত', 'দত্তগুপ্ত', 'দাশগুপ্ত', 'দাস', 'দাস শর্মা', 'দে',
	'দেব', 'নন্দী', 'নাগ', 'পাল', 'পালিত', 'বসু', 'বাঁধ', 'বাগচি', 'বৈদ্য', 'বোস', 
	'ব্যানার্জি', 'ভট্টাচার্য', 'ভট্টাচার্য', 'ভাদুড়ি', 'মিত্র', 'মুখার্জী', 'মৈত্র', 
	'রায়', 'লাভ', 'লাহিড়ী', 'সরকার', 'সান্যাল', 'সাহা', 'সিং', 'সিনহা', 'সেন', 
	'সেন-শর্মা', 'সেনগুপ্ত', 'সোম']

bengali_firstnames_female = ['Adrija', 'Ananya', 'Anindita', 'Ankita', 
	'Anuradha', 'Anushka', 'Anwesha', 'Aparna', 'Arijita', 'Arundhuti', 
	'Asmita', 'Bipasha', 'Bishakha', 'Chaity', 'Chandrayee', 'Debanjana', 
	'Debapriya', 'Debarati', 'Debasmita', 'Durba', 'Durga', 'Geeta', 'Indrani', 
	'Ipshita', 'Ishita', 'Jyoti', 'Kamala', 'Kamalika', 'Lakshmi', 'Madhuparna',
	'Mohar', 'Moumita', 'Nabanita', 'Naireeta', 'Nayan', 'Nayanika', 'Nikita', 
	'Nivedita', 'Pallabi', 'Pallavi', 'Paloma', 'Pampa', 'Paromita', 'Payal', 
	'Piyali', 'Prerona', 'Priya', 'Priyanka', 'Radhika', 'Reema', 'Reshma', 
	'Rituparna', 'Riya', 'Rohini', 'Roshni', 'Ruma', 'Rumela', 'Rupsa', 
	'Sanghamitra', 'Sataraupa', 'Sayani', 'Sayantani', 'Shalini', 'Shayoni', 
	'Shreya', 'Shweta', 'Sreemoyee', 'Subha', 'Sudarshana', 'Sudeshna', 
	'Sudipta', 'Suparna', 'Sushmita', 'Swagata', 'Tanurina', 'Tanya', 'Tista', 
	'Uma', 'Upasana', 'Varsha']

bengali_firstnames_male = ['Abhijit', 'Abhishek', 'Aditya', 'Agniva', 'Alok', 
	'Amit', 'Amitava', 'Ananyo', 'Aniruddha', 'Ankur', 'Arghya', 'Arijit', 
	'Arindam', 'Aritra', 'Arka', 'Arko', 'Avik', 'Avishek', 'Ayan', 'Bhaskar', 
	'Bikash', 'Bishwadeep', 'Chandan', 'Debajyoti', 'Deeptiman', 'Dhrubo', 
	'Dipankar', 'Dipayan', 'Ganesh', 'Gaurab', 'Gaurav', 'Gautam', 'Gopal', 
	'Himadri', 'Indrajit', 'Indranil', 'Jayanta', 'Jishnu', 'Kuntal', 'Milan', 
	'Mithun', 'Monoranjan', 'Mukul', 'Niladri', 'Pankaj', 'Prasenjit', 
	'Praveen', 'Preetam', 'Raghav', 'Rahul', 'Raja', 'Rajat', 'Ranajoy', 
	'Ratan', 'Ritam', 'Sabyasachi', 'Saikat', 'Samrat', 'Sandeep', 'Sandip', 
	'Sanjay', 'Sankalpa', 'Saptarshi', 'Sayan', 'Shayok', 'Siddhartha', 'Soham',
	'Somnath', 'Soumya', 'Souparna', 'Sourabh', 'Sourav', 'Sourojit', 'Souvik', 
	'Subhashish', 'Subrata', 'Sudipto', 'Sukumar', 'Sumit', 'Sunny', 'Swagato', 
	'Tapan', 'Tapas', 'Tathagata', 'Tushar', 'Udayan', 'Utsab']

#male and female

urdu_firstnames = ["Azhar", "Benazir", "Farahnaz", "Maliha", "Minoo", "Romana", "Sania", "Azhar", "Burhan", "Changezi", "Faizan", "Fasih", "Fuad", "Hassim", "Jan", "Shoaib", "ازہر", "بے نظیر", "فرحناز", "ملیحہ", "مینو", "رومانہ", "ثانیہ", "ازہر", "برہان", "تبدیلی", "فیضان", "فسیح", "فود", "حسم", "جان", "شعیب"]
urdu_surnames = ["Abid", "Ahmad", "Akbar", "Akmal", "Alam", "Ayaz", "Bohra", "Bucha", "Bukhari", "Buksh", "Bux", "Chandpuri", "Changezi", "Emani", "Farrukhabadi", "Farrukhi", "Fazail", "Hassim", "Hilaly", "Hussaini ", "Brahmin", "Lucknawi", "Ludhianvi", "Matloob", "Omar", "Vaishya", "Rahimtoola", "Shafiq", "Shoaib", "Siddiqui", "Siddiqui", "Tikka", "Yasser", "عابد", "احمد", "اکبر", "اکمل", "عالم", "ایاز", "بوہرہ", "بوچا", "بخاری", "بخش", "بک", "چاندپوری", "چینزی", "ایمانی", "فرخ آبادی", "فرخی", "فضل", "حصیم", "ہلالی", "حسینی", "برہمن", "لکنوئی", "لدھیانوی", "متلوب", "عمر", "واشیا", "رحیمتولہ", "شفیق", "شعیب", "صدیقی", "صدیقی", "ٹکا", "یاسر"]

#basque and catalan - use Spanish names for now
