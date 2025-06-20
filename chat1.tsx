import React, { useState, useMemo } from 'react';
import { ChatMessage } from '../types';
import { FileText, User, Bot, Star, Users, Package, MapPinned, Filter, ChevronDown, BadgePercent } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ChatBubbleProps {
  message: ChatMessage;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isUser = message.type === 'user';
  const hasAttachments = message.attachments && message.attachments.length > 0;
  
  // Filter states
  const [priceSort, setPriceSort] = useState<'none' | 'low-high' | 'high-low'>('none');
  const [ratingSort, setRatingSort] = useState<'none' | 'high-low' | 'low-high'>('none');
  const [areaFilter, setAreaFilter] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [showAllAmenities, setShowAllAmenities] = useState<Record<number, boolean>>({});
    
  const extractWorkspaceRecommendations = (text: string) => {
    const recommendations: any[] = [];
    const lines = text.split('\n');
    let currentRec: any = {};

    lines.forEach(line => {
      if (line.match(/^\d+\./)) {
        if (Object.keys(currentRec).length > 0) {
          recommendations.push(currentRec);
        }
        // Extract name and area if present in the format: 1. Workspace Name (Area)
        const nameAreaMatch = line.match(/^\d+\.\s+([^()]+?)(?:\s*\(([^)]+)\))?$/);
        if (nameAreaMatch) {
          currentRec = { name: nameAreaMatch[1].trim() };
          if (nameAreaMatch[2]) {
            currentRec.area = nameAreaMatch[2].trim();
          }
        } else {
          currentRec = { name: line.replace(/^\d+\.\s+/, '') };
        }
      } else if (currentRec.name) {
        const addressMatch = line.match(/Address:\s+(.*)/);
        const typeMatch = line.match(/Workspace Type:\s+(.*)/);
        const offeringsMatch = line.match(/Offerings:\s+(.*)/);
        const amenitiesMatch = line.match(/Amenities:\s+(.*)/);
        const seatsMatch = line.match(/Seats Available:\s+(.*)/);
        const ratingMatch = line.match(/Rating:\s+(.*)/);
        const categoryMatch = line.match(/Category:\s+(.*)/);
        const priceMatch = line.match(/Price:\s+₹(.*)/);
        const similarityMatch = line.match(/Similarity Score:\s+([\d.]+)%/);
        const linkMatch = line.match(/Link:\s+\[(.*?)\]\((.*?)\)/);

        if (addressMatch) currentRec.address = addressMatch[1];
        if (typeMatch) currentRec.workspace_type = typeMatch[1];
        if (offeringsMatch) currentRec.offerings = offeringsMatch[1];
        if (amenitiesMatch) currentRec.amenities = amenitiesMatch[1];
        if (seatsMatch) currentRec.seats = seatsMatch[1];
        if (ratingMatch) currentRec.rating = parseFloat(ratingMatch[1]);
        if (categoryMatch) currentRec.category = categoryMatch[1];
        if (priceMatch) currentRec.price = parseInt(priceMatch[1]);
        if (similarityMatch) currentRec.similarity_score = parseFloat(similarityMatch[1]);
        if (linkMatch) {
          currentRec.linkText = linkMatch[1];
          currentRec.linkUrl = linkMatch[2];
        }
      }
    });

    if (Object.keys(currentRec).length > 0) {
      recommendations.push(currentRec);
    }

    return recommendations;
  };

  let recommendationsTextToUse = message.content;
  const workspaceRecommendations = !isUser ? extractWorkspaceRecommendations(recommendationsTextToUse) : [];
  const hasRecommendations = workspaceRecommendations.length > 0;

  // Check if user has requested filters in their message or if the assistant mentions filters
  const hasFilterRequest = !isUser && (
    message.content.toLowerCase().includes('filter') ||
    message.content.toLowerCase().includes('sort') ||
    message.content.toLowerCase().includes('show filters') ||
    message.content.toLowerCase().includes('filtering options')
  );

  // Enhanced area extraction function
  const extractAreaFromWorkspace = (workspace: any): string[] => {
    const areas: string[] = [];
    
    // First try to get area from the workspace object directly
    if (workspace.area && typeof workspace.area === 'string' && workspace.area.trim().length > 0) {
      areas.push(workspace.area.trim().toLowerCase());
    }
    
    // If no area found, try to extract from address
    if (areas.length === 0 && workspace.address) {
      const address = workspace.address.toLowerCase();
      const addressParts = address.split(',').map((part: string) => part.trim());
      
      // Look for area indicators in address parts
      const areaIndicators = ['sector', 'phase', 'block', 'area', 'nagar', 'colony', 'extension', 'park'];
      const excludeTerms = ['road', 'street', 'avenue', 'lane', 'building', 'floor', 'office', 'tower', 'complex'];
      
      for (const part of addressParts) {
        // Skip if it contains exclude terms
        if (excludeTerms.some(term => part.includes(term))) continue;
        
        // Check if it contains area indicators or looks like an area name
        if (areaIndicators.some(indicator => part.includes(indicator)) || 
            (part.length > 3 && part.length < 30 && !part.match(/\d+/))) {
          areas.push(part);
          break; // Take the first valid area found
        }
      }
    }
    
    return areas;
  };

  // Extract unique areas for the current city from all recommendations
  const allAreas = useMemo(() => {
    const areaSet = new Set<string>();
    const allRecs = !isUser ? extractWorkspaceRecommendations(recommendationsTextToUse) : [];
    allRecs.forEach(workspace => {
      const areas = extractAreaFromWorkspace(workspace);
      areas.forEach(area => areaSet.add(area));
    });
    return Array.from(areaSet).sort();
  }, [recommendationsTextToUse, isUser]);

  // Apply filters and sorting
  const filteredAndSortedRecommendations = useMemo(() => {
    let filtered = [...workspaceRecommendations];

    // Apply area filter
    if (areaFilter !== 'all') {
      filtered = filtered.filter(workspace => {
        const areas = extractAreaFromWorkspace(workspace);
        return areas.some(area => area.trim().toLowerCase() === areaFilter.trim().toLowerCase());
      });
    }

    // Only show workspaces with a similarity score (any value, not undefined)
    filtered = filtered.filter(workspace => typeof workspace.similarity_score !== 'undefined' && workspace.similarity_score > 75);

    // Apply sorting
    if (priceSort !== 'none') {
      filtered.sort((a, b) => {
        const priceA = a.price || 0;
        const priceB = b.price || 0;
        return priceSort === 'low-high' ? priceA - priceB : priceB - priceA;
      });
    } else if (ratingSort !== 'none') {
      filtered.sort((a, b) => {
        const ratingA = a.rating || 0;
        const ratingB = b.rating || 0;
        return ratingSort === 'high-low' ? ratingB - ratingA : ratingA - ratingB;
      });
    }

    return filtered;
  }, [workspaceRecommendations, priceSort, ratingSort, areaFilter]);

  const [introText = '', recommendationsText = ''] = (recommendationsTextToUse || '').split('\n\nHere are some workspace recommendations for you:');

  // Check if similarity scores are present in the recommendations text
  const hasSimilarityScores = recommendationsText && recommendationsText.includes('Similarity Score:');

  // Only show introText if it is not empty, not just whitespace, and not just a code block or Gemini JSON
  let showIntro = false;
  let introToShow = '';
  if (typeof introText !== 'undefined' && introText.trim().length > 0) {
    // Remove code blocks and Gemini JSON from introText
    const cleanedIntro = introText.replace(/```[\s\S]*?```/g, '').replace(/\{[\s\S]*?\}/g, '').trim();
    if (cleanedIntro.length > 0) {
      showIntro = true;
      introToShow = cleanedIntro;
    }
  } else if (!hasRecommendations && message.content.trim().length > 0) {
    showIntro = true;
    introToShow = message.content;
  }

  const resetFilters = () => {
    setPriceSort('none');
    setRatingSort('none');
    setAreaFilter('all');
  };

  const hasActiveFilters = priceSort !== 'none' || ratingSort !== 'none' || areaFilter !== 'all';
  
  // Ensure timestamp is a Date object
  let timestamp: Date;
  if (message.timestamp instanceof Date) {
    timestamp = message.timestamp;
  } else if (typeof message.timestamp === 'string') {
    timestamp = new Date(message.timestamp);
  } else {
    timestamp = new Date();
  }

  return (
    <div className="flex flex-col space-y-4 animate-fadeIn">
      {/* Main message bubble */}
      {showIntro && (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
          <div className={`flex items-start max-w-[80%] md:max-w-[70%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
            <div className={`flex-shrink-0 rounded-full p-2 ${isUser ? 'bg-blue-500 ml-2' : 'bg-teal-600 mr-2'}`}>
              {isUser ? (
                <User size={16} className="text-white" />
              ) : (
                <Bot size={16} className="text-white" />
              )}
            </div>
            <div className={`${
              isUser
                ? 'bg-blue-500 text-white rounded-2xl rounded-tr-none'
                : 'bg-white text-black rounded-2xl rounded-tl-none'
            } py-3 px-4 shadow-sm`}>
              <div className="text-sm md:text-base whitespace-pre-wrap break-words">
                <ReactMarkdown>{introToShow}</ReactMarkdown>
              </div>
              {hasAttachments && (
                <div className="mt-2 pt-2 border-t border-opacity-20 border-gray-200">
                  {message.attachments?.map((file, index) => (
                    <div key={index} className="flex items-center text-xs">
                      <FileText size={14} className={isUser ? 'text-blue-100' : 'text-blue-500'} />
                      <span className="ml-1 truncate max-w-[200px]">{file.name}</span>
                    </div>
                  ))}
                </div>
              )}
              <div className={`text-xs mt-1 ${isUser ? 'text-gray-200' : 'text-gray-400'}`}>
                {timestamp.toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Filter Controls - Only show if user has requested filters */}
      {hasRecommendations && hasFilterRequest && (
        <div className="ml-12">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-2 px-3 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
              >
                <Filter size={16} className="text-gray-500" />
                Filters
                <ChevronDown 
                  size={16} 
                  className={`text-gray-500 transition-transform ${showFilters ? 'rotate-180' : ''}`} 
                />
              </button>
              {hasActiveFilters && (
                <button
                  onClick={resetFilters}
                  className="px-3 py-2 text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  Clear All
                </button>
              )}
            </div>
            <div className="text-sm text-gray-600">
              {filteredAndSortedRecommendations.length} of {workspaceRecommendations.length} workspaces
            </div>
          </div>
          
          {showFilters && (
            <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4 shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Price Sort */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sort by Price
                  </label>
                  <select
                    value={priceSort}
                    onChange={(e) => {
                      setPriceSort(e.target.value as any);
                      if (e.target.value !== 'none') setRatingSort('none');
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  >
                    <option value="none">Default</option>
                    <option value="low-high">Price: Low to High</option>
                    <option value="high-low">Price: High to Low</option>
                  </select>
                </div>

                {/* Rating Sort */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sort by Rating
                  </label>
                  <select
                    value={ratingSort}
                    onChange={(e) => {
                      setRatingSort(e.target.value as any);
                      if (e.target.value !== 'none') setPriceSort('none');
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  >
                    <option value="none">Default</option>
                    <option value="high-low">Rating: High to Low</option>
                    <option value="low-high">Rating: Low to High</option>
                  </select>
                </div>

                {/* Area Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Filter by Location
                  </label>
                  <select
                    value={areaFilter}
                    onChange={(e) => setAreaFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    disabled={allAreas.length === 0}
                  >
                    <option value="all">All Locations</option>
                    {allAreas.map((area) => (
                      <option key={area} value={area}>
                        {area.charAt(0).toUpperCase() + area.slice(1)}
                      </option>
                    ))}
                  </select>
                  {allAreas.length === 0 && (
                    <p className="text-xs text-gray-500 mt-1">No location information available</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Workspace recommendations cards */}
      {hasRecommendations && (
        <div className="ml-12 overflow-x-auto pb-4">
          <div className="flex gap-4">
            {(hasFilterRequest ? filteredAndSortedRecommendations : workspaceRecommendations.filter(w => !w.similarity_score || w.similarity_score > 75)).map((workspace, index) => (
              <div
                key={index}
                className="flex-shrink-0 w-72 bg-white rounded-lg overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow flex flex-col relative"
              >
                {/* Similarity Score Badge: Only show if similarity scores are present and this workspace has one */}
                {hasSimilarityScores && workspace.similarity_score && Number(workspace.similarity_score) > 75 && (
                  <div
                    className="absolute top-3 right-3 bg-gradient-to-r from-green-400 to-blue-500 text-white text-xs font-bold px-3 py-1 rounded-full shadow-md flex items-center z-10 animate-fadeIn"
                    title={`Similarity Score: ${workspace.similarity_score}%`}
                  >
                    <BadgePercent size={14} className="mr-1 text-yellow-300" />
                    {workspace.similarity_score}%
                  </div>
                )}
                
                <div className="flex flex-col h-full p-4">
                  <h3 className="font-semibold text-gray-900 mb-1 pr-16">
                    {workspace.name}
                  </h3>
                  <div className="flex items-start text-sm text-gray-600 mb-3">
                    <MapPinned size={14} className="mr-1 mt-0.5 flex-shrink-0" />
                    <span className="line-clamp-2">{workspace.address}</span>
                  </div>
                  
                  <div className="space-y-2 mb-3">
                    <div className="flex items-center text-sm">
                      <Package size={14} className="mr-2 text-gray-500" />
                      <span className="text-gray-700">{workspace.workspace_type}</span>
                    </div> 
                    <div className="flex items-center text-sm">
                      <Users size={14} className="mr-2 text-gray-500" />
                      <span className="text-gray-700">{workspace.seats} seats</span>
                    </div>
                    <div className="flex items-center text-sm">
                      <Star size={14} className="mr-2 text-yellow-500" />
                      <span className="text-gray-700">{workspace.rating} rating</span>
                    </div>
                  </div>
                  
                  <div className="text-sm text-gray-600 mb-3">
                    <strong>Category:</strong> {workspace.category}
                  </div>
                  
                  <div className="text-sm text-gray-600 mb-3">
                    <strong>Offerings:</strong> {workspace.offerings}
                  </div>
                  
                  {workspace.amenities && (
                    <div className="text-sm text-gray-600 mb-3">
                      <strong>Amenities:</strong>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {(() => {
                          const amenitiesArr = workspace.amenities.split(', ');
                          const isExpanded = !!showAllAmenities[index];
                          const visibleAmenities = isExpanded ? amenitiesArr : amenitiesArr.slice(0, 6);
                          const hiddenCount = amenitiesArr.length - 6;

                          return (
                            <>
                              {visibleAmenities.map((amenity: string, i: number) => (
                                <span
                                  key={i}
                                  className="inline-block bg-gray-100 rounded px-2 py-0.5 text-xs"
                                >
                                  {amenity}
                                </span>
                              ))}
                              {hiddenCount > 0 && !isExpanded && (
                                <button
                                  className="inline-block bg-gray-200 rounded px-2 py-0.5 text-xs text-blue-600 hover:underline"
                                  onClick={() => setShowAllAmenities(prev => ({ ...prev, [index]: true }))}
                                >
                                  +{hiddenCount} more
                                </button>
                              )}
                              {isExpanded && amenitiesArr.length > 6 && (
                                <button
                                  className="inline-block bg-gray-200 rounded px-2 py-0.5 text-xs text-blue-600 hover:underline"
                                  onClick={() => setShowAllAmenities(prev => ({ ...prev, [index]: false }))}
                                >
                                  Show less
                                </button>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  )}

                  
                  <div className="mt-auto flex items-center justify-between pt-4">
                    <div className="text-lg font-semibold text-gray-900">
                      ₹{workspace.price}
                    </div>
                    <a
                      href={workspace.linkUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3 py-1.5 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 transition-colors"
                    >
                      View Details
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {hasFilterRequest && filteredAndSortedRecommendations.length === 0 && hasActiveFilters && (
            <div className="text-center py-8 text-gray-500">
              <Filter size={48} className="mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium mb-2">No workspaces found with this filter</p>
              <p className="text-sm">Try adjusting your filter criteria or clearing all filters.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ChatBubble;