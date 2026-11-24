# Nearest real training review to generated vectors

Cosine similarity in the standardised CLS space; label = Workshop #2's prediction for the generated vector. For "real", the 16 vectors are validation reviews (the similarity a genuine unseen review gets).

## Real (validation)

| # | Predicted | Cosine | Nearest training review (truncated) |
| --- | --- | --- | --- |
| 1 | positive | 0.73 | Disneyland has been around for as long as I can remember and it still has the same feel although different. I love the safe feeling we have while inside the park and how clean everything looks to be.… |
| 2 | positive | 0.81 | From the moment we stepped through the doors of Disneyland we were captivated and entranced by it.It took us back to our childhood with familiar characters we all love to see.The rides and exhibits were amazing.Not to be… |
| 3 | positive | 0.84 | I hear all the time that Paris Disney land isn   t as good as America and that may be so I can not compare as I haven   t been there but I can say that we had an amazing time here . We have two children with disabilities… |
| 4 | positive | 0.63 | The magic of Disney did not disappoint. Visited the week before Christmas (unfortunately two overcast days with light rain). Lots to see and do. Food was reasonably priced (to be expected)   noting there was no restricti… |
| 5 | positive | 0.87 | We spent two days over Spring Break at Disneyland with our 4 year old (almost 5) son and 6 month old daughter, and had an absolute blast. My wife and I visited as a couple more than a decade ago, and the park is now even… |
| 6 | neutral | 0.65 | Went there with our two kids (one was 4 and one was 2 ) in July. It was great to catch the subway to Disneyland. My 4 year old was very excited to see Mickey mouse' ears on the train. We only got the one day tickets. But… |
| 7 | positive | 0.81 | Disneyland Paris is my favourite place! I absouelty love the feeling you get while walking around the parks, everything feels so magical! The rides are great and the character interactions are amazing! Disney is a great … |
| 8 | negative | 0.90 | My kids (7 & 5) have asked continously for a trip to Disneyland for a long a time now. They were over the moon when we surprised them with a visit this Jan 2 and 3. But halfway through the first day they declared it was … |
| 9 | negative | 0.94 | Over Priced Over Crowed,Under whelming Under performing !!Way $$Expensive with Crappy overpriced gifts,little of 'Free' performances  All rides take at least an hour to get on and 'Future Land' mose well be called 'Star … |
| 10 | positive | 0.79 | As a single Mum, I knew that if I wanted to save for a holiday, it had to be one to remember, so I booked Disneyland Paris and revealed it to my daughter for her birthday. We went over Halloween and honestly, never thoug… |
| 11 | negative | 0.93 | Having been to Disney before I was so looking forward to it but once arriving I was to be in for a major shock the ques were terrible 70 mins plus for rides even with ur fast past it's not much better waiting to see char… |
| 12 | neutral | 0.86 | I recently reviewed Universal Studios in Singapore, and I just thought I'd review the one that I compared USS with. I went here before I even went to Universal Studios, and I found it more cheerful, to say the least, to … |
| 13 | positive | 0.82 | This is our second visit to the park and what a wonderful trip we had. So much to do and see. The theme is Springtime and the park is looking wonderful with flowers galore!The kids love it and so do we. It's great to see… |
| 14 | negative | 0.96 | We had planned to have the entire day in the park and thought that there would be lots to do but by around midday we were all bored and wanted to go back to the hotel. The rides were extremely dated and each of them had … |
| 15 | positive | 0.81 | Wow! So spectacular, I was blown away. This is a must see destination that I wish everyone could have the opportunity to experience in their lifetime! Everything was just perfect! To top off the day the most amazing para… |
| 16 | positive | 0.78 | We had a wonderful time. Disneyland is truly the happiest place on earth. The shows were excellent. The staff are friendly and helpful.… |

## Vanilla GAN

| # | Predicted | Cosine | Nearest training review (truncated) |
| --- | --- | --- | --- |
| 1 | positive | 0.74 | excellent service good quality  nice management  very good Disneyland resort  nice public transport facility… |
| 2 | neutral | 0.82 | If this is your first trip to Disney you will enjoy this place. If you have been to Florida La you will be disappointed as so small in comparison. Really easy to get to. Mtr takes you to the door. Tickets reasonable. We … |
| 3 | positive | 0.80 | We did Disneyland for the first time on the 24th, 25th and 27th December. It was a great experience. In the three days (more on the first day) we did most of the main rides. Loved Space mountain, Star tours, Big thunder … |
| 4 | positive | 0.46 | No matter what age you are this is a great day so go and enjoy the rides.  Most are aimed at younger visitors some are worth a repeat ride… |
| 5 | positive | 0.70 | If you havent been to HK Disneyland, its all exciting and fun. But usually its not a place for return visit unless after the new Iron Man Ride is out. Don't go during school holidays and public holidays if you want to av… |
| 6 | positive | 0.64 | We have been here a few years ago. This time it was on a Tues. Wed. Almost no lines for anything we wanted to do or see. Only sort of long line was for Autopia    but after we waited for 10 minute, the ride cancelled due… |
| 7 | positive | 0.78 | Disney does so many things well. We went in early May on a Thursday and Friday. The parks were busy both days, but Disneyland Park was packed on Friday. Since we had little kids, we never waited too long in line, both be… |
| 8 | positive | 0.85 | This really is the happiest place on earth! Loved my day here had a smile on my face through out the day. The staff are so friendly and cheerful. Young and old are treated exactly the same. This place is a must no matter… |
| 9 | positive | 0.66 | This Disneyland is tiny but perfect for young children. Not too crowded compared to Tokyo. Had a nice visit there in Dec 2015.… |
| 10 | negative | 0.89 | We had travelled to HK with a visit to Disneyland with our children and niece (aged 9 to 15) as a priority. In summary, the tickets were expensive for the amenities, the park was small with limited rides, one of which wa… |
| 11 | neutral | 0.80 | Rides are great but some staff (especially the ones at the train rounding disneyland) are rude towards the guests.… |
| 12 | positive | 0.75 | Growing up in Orlando, FL and visiting Disney World many times, I was so excited to see Disneyland. The look and feel of Main Street was as magical as the FL park. We were pleasantly surprised at the sutble differences o… |
| 13 | positive | 0.79 | Everything is fantastic, it really is the happiest place on earth! The staff are incredible, enthousiastic, welcoming, helpful, I could go on and on!… |
| 14 | positive | 0.71 | Disneyland is one of my favorites.  Now that I can enjoy it with my grandchildren, I think it's even better. The Fast Pass really helps out with the little and older ones.  Go on a slightly rainy day and it's the best.… |
| 15 | positive | 0.79 | i've heard that we'd be better off going to ocean park but there's something about disney you can't say no to. I'd never been to disneyland before which was also an attraction. I never got Ocean Park, but maybe we will n… |
| 16 | negative | 0.95 | This is our 3rd visit to the park and it will be our lastThe place is a shambles and a VERY expensive one at 550 euros for 4 of us for 2 daysFor that I expect ALL the rides working. Queued 3 times for Thunder Mountain on… |

## WGAN-GP

| # | Predicted | Cosine | Nearest training review (truncated) |
| --- | --- | --- | --- |
| 1 | neutral | 0.74 | Firstly they have some great rides, really enjoyed the big rides in both parks and some of the shows. Food and refreshments are very expensive probably made worse by the poor GBP Euro exchange rate. Some of the rides see… |
| 2 | positive | 0.58 | Disneyland is always so fun. But it seems like the prices are getting higher and it's getting more crowded. There was a fight that almost broke out around us because of how crowded it was. But you can't beat the magic th… |
| 3 | positive | 0.76 | You simply cannot describe your first experience visiting Disneyland Paris, I was 22 but I simply loved it and got totally wrapped up in the 'magic' there was a vast selection of rides, eateries, transport, characters an… |
| 4 | neutral | 0.69 | I have been to US and Japan's Disneyland   HKG is the smallest   even though with the latest new theme park. Certainly, it is a wonderful place for my kids as they love to visit Disneyland whenever we visit HKG.However, … |
| 5 | positive | 0.62 | The trip was magical. We had five days   four nights which was just about right for a first visit to here and the Studios. The characters were a huge attraction. Start seeking autographs from your first day. Our boys had… |
| 6 | positive | 0.75 | Who wouldn't love the happiest place on Earth. This was our 11th visit this year as we purchased season passes for the first time ever. They were worth every penny! Love the new 60th anniversary celebration especially th… |
| 7 | neutral | 0.87 | The train ride from Sunny bay to Disneyland is worth taking. There is always a huge rush from the morning. Better to reach the park at 10 am. Book your tickets in advance to skip the queue. Evening parade and light show … |
| 8 | positive | 0.82 | We had an amazing time visiting Disneyland. We purchased a 3 day park hopper MAX pass and it was worth spending the extra money to have access to both parks, fast pass and photo's. It was also great to be able to do the … |
| 9 | positive | 0.49 | Disneyland at Christmas is great and the crowd levels were good.… |
| 10 | negative | 0.93 | We travel a lot and of course this time was with our two children 8 and 3 years old. This was our 3 year olds first time to any Disney park and our son we have brought four times prior to DisneyWorld. I haven't been to D… |
| 11 | positive | 0.79 | Back on Dec 14th of this year, both me and my mother got invited down here by her boss who  1 paid for our hotel, admission and food which was a huge plus to us. Crowds were insane and I was expecting that but that didn'… |
| 12 | positive | 0.77 | Disneyland really is the happiest place on earth. My first visit to this park all the way from the UK could not have been more magical, I felt like a child again. Nothing can beat your first steps walking down Mainstreet… |
| 13 | positive | 0.77 | Here are my tips and guidelines for those who are interested in going to Disneyland:1) If possible, go there during weekend. Its just for the fireworks. Its because fireworks is a confirmed event during weekend but not a… |
| 14 | positive | 0.68 | If you like Christmas this is the perfeft time to go!The park Is amazing , huge, clean and entertaining.You have both children rides and rollacoaster for adults.If you are staying at a Disney hotel keep in mind that you … |
| 15 | negative | 0.88 | We were in Paris for 8 days with our children. Most of our time was spent exploring nooks and crannies of the city or visiting the big sites like the Orsay, Sacre Coeur, and others. But on a lark we decided to try out th… |
| 16 | positive | 0.67 | Disneyland was a pretty good experience. This was my second trip there this year. The only thing I can complain about are the ride closures and shutdowns. On both occasions I experienced ride closures which is very frust… |

## cGAN (vanilla)

| # | Predicted | Cosine | Nearest training review (truncated) |
| --- | --- | --- | --- |
| 1 | neutral | 0.83 | First off, the visit was enjoyable overall. Would I do it again? No, probably save a little longer and go to the  real  Disney in the USA. I have to say that one of the most disappointing things of all was the staff. Not… |
| 2 | positive | 0.60 | Can be expensive to get in but the staff do an overall good job of entertaining families with kids. There are a lot of strollers to contend with in the crowds, but if you have a stroller, on most rides, you can use the h… |
| 3 | positive | 0.65 | The two park set up is Excellent for first time Disney goers. The Nemo ride was fast and cool. Raiders of the lost ark roller coaster was awesome. Was upside down for a brief moment. The parachute ride was OK for smaller… |
| 4 | negative | 0.96 | We have just returned from EuroDisney and was appalled at how badly run it was   Having visited when the attraction first opened I swore that I would never go again given that the service was so bad, but as MGM had recen… |
| 5 | neutral | 0.83 | Went here with family. Took a long travel from Tsim Sha Tsui area. Around 40 minutes. Got there, if you already bought tickets online, just proceed to guest relation counter. Inside the park, if you go together with youn… |
| 6 | positive | 0.80 | What a fantastic place for all ages. Had a great time there with my grown children and their partners. You are never to old for Disney… |
| 7 | negative | 0.94 | Attended for short break at end of school holidays. Paid a lot for the break and added extra to have a uprated room in newport hotel and  full board  plus meal plan. The hotel was nice but the room it's self was shocking… |
| 8 | neutral | 0.85 | Firstly they have some great rides, really enjoyed the big rides in both parks and some of the shows. Food and refreshments are very expensive probably made worse by the poor GBP Euro exchange rate. Some of the rides see… |
| 9 | positive | 0.74 | We have visited Disney in Orlando FL, as well as Disney in Shanghai. Disney Paris was amazing! It's very similar to the Orlando's Magic Kingdom. Super Clean and there is something for everyone! Our daughters are 6 & 12  … |
| 10 | positive | 0.80 | We had an amazing trip to Disneyland!!! We spent 2 days at Disneyland and 1 day at California Adventure. Zero complaints here! Shows were great, as were the rids. The cast members are so nice and helpful! Took our kids w… |
| 11 | positive | 0.65 | I love this place, once you go through disneyland doors, your fun begins. No matter how many times you have been there before it's always a different experience, employees are nice , rides are fun, except the ride lines … |
| 12 | neutral | 0.79 | So this was i havent been here since i was 5. i enjoyed the free shows especially the fireworks and fantasia style with mikey mouse. However the rides werent much to rave about. jurrasic park was good and unfortunatley t… |
| 13 | positive | 0.54 | Me and my family really enjoy our trip here. But there not enough time because to play the ride we need to queues, and it takes a very long time. And I recommend that anyone who is going here need to buy a 2 day pass if … |
| 14 | positive | 0.86 | I took my daughter early in August, 2012. We spent an entire day at Disneyland, from opening to closing and had an absolute blast. The grounds are beautiful, the staff we met were all friendly. We went on almost everythi… |
| 15 | positive | 0.45 | Bought tickets online to save time when entering park. Spent the full day at park and closed off with night display of fireworks. The queues were a bit long but fast tracked queues from vouchers received at Disney hotel.… |
| 16 | positive | 0.89 | I first visited Disneyland when I was 10 more than 20 years ago. It was the greatest trip of my life. Now with two young children of my own, I was anxious to go back. We finally made it last week and the magic has only b… |

## cGAN (WGAN-GP)

| # | Predicted | Cosine | Nearest training review (truncated) |
| --- | --- | --- | --- |
| 1 | neutral | 0.74 | Place is nice and service as well. Kitchen poor and super bad   bloody expensive for what u get. Pumpkin soup: cold and 0 taste. Prawns cold, mashy not cleaned with excrementes still inside. Pasta for kids: ice cold, sup… |
| 2 | positive | 0.69 | We LOVE Disneyland.  Clean park, friendly staff and great food.  Always such a great family day.  We always go in February and October.  We have found that those are the least busiest times!!… |
| 3 | positive | 0.60 | The Hong Kong Disneyland is quite big, one day is not enough to explore all the places in the Disneyland. I went there on weekend i only manage to go for 4 rides only. but the experience is quite good but tired.… |
| 4 | negative | 0.93 | As an Aussie, my fiance and I visited Disneyland in Anahiem for the first time in our 20's and we LOVED it, so much so that we returned another time a year later. Im a big Disney fan girl.However, Hong Kong Disneyland wa… |
| 5 | neutral | 0.81 | Went to Disneyland Paris November 2012 with Mother and my 2,5 year old little girl. Park looks absolutely beautiful, very well constructed, very detailed perfection. But, very crowded and busy even out of the holiday sea… |
| 6 | positive | 0.61 | A heaven for kids. A true Magical experience. It is huge if u want to cover you need at least 2 full days. What we liked in attraction was Pirates of Caribbean, space ride & the show. parade was something you cant afford… |
| 7 | negative | 0.93 | Our entire trip was planned around visiting Disneyland and luckily we did enough other great things on our holiday for this not to ruin it.Where do I start.. Don't get there early if you are not staying at the park (Don'… |
| 8 | neutral | 0.82 | We stayed in the Santa Fe hotel which is basic but everything you need, bed, toilet, sink etc. you spend very little time in the rooms. Hotel a little dated but nothing to worry about. There's a phone in the room for a w… |
| 9 | positive | 0.59 | Its the most beautiful place i have ever been.Really had lots of fun in the rides.… |
| 10 | positive | 0.62 | The loving moment  Grandness of the place  Surreal in every respect  Go crazy with range of Souvenir Shops  Cool rides  Well a super happy feeling the place givesThings to Watch out:  Take the map and plan your trip. You… |
| 11 | positive | 0.86 | They don't call it the happiest place on earth for no reason! We want on an adults only trip and this would be my first visit to the park since I was 11 years old. I was simply blown away by how awesome it was! The park … |
| 12 | positive | 0.63 | Disneyland   the name says it all!  Great colourful place, especially for young ones.  Wear comfortable clothes and shoes as its a long day out.  Quite a lot of walking.… |
| 13 | positive | 0.60 | I visited all the parks in 1 day(No kids) . The disneyland los angeles has 2 main parks 1. The disneyland california adventure 2. The disneyland park, With parkhooper tickets you can visit both the parks in one day. Cali… |
| 14 | positive | 0.57 | I have been to Disney world in Florida 32 times in 31 years. My husband and I got married there 9 years ago. To say I'm a disney pro is an understatement. For my most recent trip to Paris I asked my husband if he wanted … |
| 15 | neutral | 0.82 | I feel dissappointed. I have visited Disney at Orlando and thought this will be similar, though smaller. I found out that this is a lovely park, with nice buildings and a truly Disney atmosphere, but there were not great… |
| 16 | positive | 0.55 | Always first class productions. My only complaint would be near the Star Tours ride the walk areas have been choked down to much and it is very hard to navigate through that area.… |
